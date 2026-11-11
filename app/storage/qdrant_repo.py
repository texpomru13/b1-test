"""Qdrant chunk repository: one collection, named dense "dense" + sparse "bm25"
with IDF (verified in spike S2). Bootstrap refuses mismatched schemas and NEVER
deletes existing data (controlled reindex is a separate explicit operation).
Worker-side operations are sync (QdrantClient); the API keeps the async client.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from app.retrieval.bm25 import SparseVector

DENSE_NAME = "dense"
SPARSE_NAME = "bm25"

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "ingestion_run_id": models.PayloadSchemaType.KEYWORD,
    "index_fingerprint": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "is_document_head": models.PayloadSchemaType.BOOL,
    "run_complete": models.PayloadSchemaType.BOOL,
}


class SchemaMismatchError(Exception):
    pass


class FingerprintMismatchError(Exception):
    def __init__(self, existing: str, computed: str) -> None:
        super().__init__(
            f"existing index fingerprint {existing[:12]}… does not match the computed "
            f"{computed[:12]}…; an explicit controlled reindex is required"
        )


def point_id(doc_id: str, run_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"b1://{doc_id}/{run_id}/{chunk_index}"))


def doc_run_filter(doc_id: str, run_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(
                key="ingestion_run_id", match=models.MatchValue(value=run_id)
            ),
        ]
    )


def create_qdrant_client(url: str, *, timeout_s: float = 5.0) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url, timeout=int(timeout_s))


def create_sync_qdrant_client(url: str, *, timeout_s: float = 30.0) -> QdrantClient:
    return QdrantClient(url=url, timeout=int(timeout_s))


class QdrantChunkRepository:
    """Async side (API): connectivity/readiness checks."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    async def ping(self) -> bool:
        await self._client.get_collections()
        return True

    async def collection_exists(self) -> bool:
        return bool(await self._client.collection_exists(self._collection_name))

    async def aclose(self) -> None:
        await self._client.close()


class QdrantIndexRepository:
    """Sync side (worker): bootstrap, upserts, verification, publication marker."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self.collection_name = collection_name

    def ensure_collection(self, *, dense_dim: int, computed_fingerprint: str) -> None:
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_NAME: models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            for field, schema in PAYLOAD_INDEXES.items():
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            return
        info = self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or DENSE_NAME not in vectors:
            raise SchemaMismatchError(
                f"collection '{self.collection_name}' lacks the named dense vector "
                f"'{DENSE_NAME}'"
            )
        dense = vectors[DENSE_NAME]
        if int(dense.size) != dense_dim or dense.distance != models.Distance.COSINE:
            raise SchemaMismatchError(
                f"collection dense vector is size={dense.size}/{dense.distance}, "
                f"required size={dense_dim}/Cosine; existing data is NOT deleted — "
                "run a controlled reindex explicitly"
            )
        sparse = info.config.params.sparse_vectors or {}
        sparse_params = sparse.get(SPARSE_NAME)
        if sparse_params is None or sparse_params.modifier != models.Modifier.IDF:
            raise SchemaMismatchError(
                f"collection sparse vector '{SPARSE_NAME}' with IDF modifier is missing"
            )
        existing_fp = self._any_existing_fingerprint()
        if existing_fp is not None and existing_fp != computed_fingerprint:
            raise FingerprintMismatchError(existing_fp, computed_fingerprint)

    def _any_existing_fingerprint(self) -> str | None:
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=["index_fingerprint"],
            with_vectors=False,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        value = payload.get("index_fingerprint")
        return str(value) if value else None

    def upsert_chunk_batch(
        self,
        *,
        doc_id: str,
        run_id: str,
        batch: list[dict[str, Any]],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        points = [
            models.PointStruct(
                id=point_id(doc_id, run_id, payload["chunk_index"]),
                vector={
                    DENSE_NAME: dense_vectors[i],
                    SPARSE_NAME: models.SparseVector(
                        indices=sparse_vectors[i].indices, values=sparse_vectors[i].values
                    ),
                },
                payload=payload,
            )
            for i, payload in enumerate(batch)
        ]
        result = self._client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )
        if result.status != models.UpdateStatus.COMPLETED:
            raise RuntimeError(f"upsert batch not completed: {result.status}")

    def count_run(self, doc_id: str, run_id: str) -> int:
        return int(
            self._client.count(
                collection_name=self.collection_name,
                count_filter=doc_run_filter(doc_id, run_id),
                exact=True,
            ).count
        )

    def mark_run_complete_on_head(self, doc_id: str, run_id: str) -> None:
        self._client.set_payload(
            collection_name=self.collection_name,
            payload={"run_complete": True},
            points=[point_id(doc_id, run_id, 0)],
            wait=True,
        )

    def delete_run(self, doc_id: str, run_id: str) -> None:
        """Delete ONLY the given (doc_id AND run_id) — never all doc points."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=doc_run_filter(doc_id, run_id)),
            wait=True,
        )

    def head_is_complete(self, doc_id: str, run_id: str) -> bool:
        head = self.retrieve_head(doc_id, run_id)
        if head is None:
            return False
        return bool((head.payload or {}).get("run_complete") is True)

    def delete_other_runs(self, doc_id: str, keep_run_id: str) -> None:
        """Remove points of NON-accepted runs for this doc only (run-scoped)."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))],
                must_not=[models.FieldCondition(
                    key="ingestion_run_id",
                    match=models.MatchValue(value=keep_run_id))],
            )),
            wait=True,
        )

    def count_other_runs(self, doc_id: str, keep_run_id: str) -> int:
        return int(self._client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))],
                must_not=[models.FieldCondition(
                    key="ingestion_run_id",
                    match=models.MatchValue(value=keep_run_id))],
            ),
            exact=True,
        ).count)

    def retrieve_head(self, doc_id: str, run_id: str) -> Any | None:
        found = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id(doc_id, run_id, 0)],
            with_payload=True,
            with_vectors=True,
        )
        return found[0] if found else None
