"""Write dense and BM25 vectors to Qdrant for a single indexing attempt.

After verifying the chunk count, mark the first chunk as complete. The worker
can then publish this run ID in Redis."""

from __future__ import annotations

import functools
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.domain.index_fingerprint import fingerprint_for_embedding
from app.domain.job import document_title
from app.inference.client import LocalModelsError, LocalModelsSyncClient
from app.inference.contract import EmbedRequest, InputType
from app.jobs.lease import LEASE_TTL_MS
from app.observability.logging import get_logger
from app.reliability.policy import retry_sync
from app.retrieval.bm25 import Bm25Adapter
from app.storage.job_lifecycle_sync import LeaseLostError, SyncJobLifecycle
from app.storage.qdrant_repo import (
    FingerprintMismatchError,
    QdrantIndexRepository,
    SchemaMismatchError,
)
from app.storage.redis_repo import utc_now_iso

logger = get_logger("jobs.indexer")

UPSERT_BATCH = 64

ERROR_EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
ERROR_INDEX_SCHEMA_MISMATCH = "INDEX_SCHEMA_MISMATCH"
ERROR_INDEX_FINGERPRINT_MISMATCH = "INDEX_FINGERPRINT_MISMATCH"
ERROR_INDEX_VERIFY_FAILED = "INDEX_VERIFY_FAILED"
ERROR_LEASE_LOST = "LEASE_LOST"
ERROR_PUBLISH_FAILED = "PUBLISH_FAILED"
ERROR_QDRANT_UNAVAILABLE = "QDRANT_UNAVAILABLE"


class IndexingError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool,
                 run_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.run_id = run_id  # the incomplete run needing compensation (if any)


@dataclass(frozen=True)
class IndexOutcome:
    run_id: str
    fingerprint: str
    chunk_count: int


def _chunk_payload(
    chunk: dict[str, Any],
    *,
    job: dict[str, Any],
    run_id: str,
    fingerprint: str,
    page_count: int,
    chunk_count: int,
    created_at: str,
) -> dict[str, Any]:
    return {
        "doc_id": job["doc_id"],
        "doc_title": document_title(job),
        "chunk_index": chunk["index"],
        "page_from": chunk["page_from"],
        "page_to": chunk["page_to"],
        "section": chunk["section"],
        "text": chunk["text"],
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "ingestion_run_id": run_id,
        "index_fingerprint": fingerprint,
        "source_sha256": job.get("file_sha256", ""),
        "created_at": created_at,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "is_document_head": chunk["index"] == 0,  # the REAL first text chunk
        "run_complete": False,  # flipped on the head after full verification
        "segments": chunk.get("segments", []),
    }


def run_indexing(
    *,
    job: dict[str, Any],
    job_id: str,
    lifecycle: SyncJobLifecycle,
    lease_token: str,
    index_repo: QdrantIndexRepository,
    embedder: LocalModelsSyncClient,
    bm25: Bm25Adapter,
    settings: Settings,
) -> IndexOutcome:
    doc_dir = Path(settings.documents_dir) / job["doc_id"]
    artifact_path = doc_dir / "parse" / "parsed.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = artifact["chunks"]
    if not chunks:
        raise IndexingError(
            ERROR_INDEX_VERIFY_FAILED, "parse produced zero chunks", retryable=False
        )

    # Dimension and identity from a REAL encode before touching the collection.
    try:
        probe = embedder.embed(
            EmbedRequest(texts=["probe"], input_type=InputType.DOCUMENT)
        )
    except LocalModelsError as exc:
        raise IndexingError(
            ERROR_EMBEDDING_UNAVAILABLE,
            f"local-models embedding service unavailable: {exc}",
            retryable=True,
        ) from exc

    fingerprint = fingerprint_for_embedding(
        probe, settings, length_fn_id=artifact["length_fn"],
        chunk_tokens=artifact["chunk_tokens"], overlap_tokens=artifact["overlap_tokens"],
    )

    def q_retry[R](op_name: str, fn: Callable[[], R]) -> R:
        # Retry transient Qdrant errors. Schema and fingerprint mismatches are permanent.
        return retry_sync(fn, op_name=op_name,
                          max_attempts=settings.qdrant_max_attempts)

    try:
        q_retry("qdrant.ensure_collection", lambda: index_repo.ensure_collection(
            dense_dim=probe.dim, computed_fingerprint=fingerprint
        ))
    except SchemaMismatchError as exc:
        raise IndexingError(ERROR_INDEX_SCHEMA_MISMATCH, str(exc), retryable=False) from exc
    except FingerprintMismatchError as exc:
        raise IndexingError(
            ERROR_INDEX_FINGERPRINT_MISMATCH, str(exc), retryable=False
        ) from exc
    except Exception as exc:  # Qdrant unreachable after bounded retries
        # Nothing was written for this attempt yet (no run id): report the real
        # cause instead of leaving the job in `running` for the reconciler.
        raise IndexingError(
            ERROR_QDRANT_UNAVAILABLE,
            f"vector store unavailable: {type(exc).__name__}: {exc}",
            retryable=True,
        ) from exc

    run_id = uuid.uuid4().hex
    created_at = utc_now_iso()
    lifecycle.set_progress(job_id, {
        "stage": "embed",
        "ingestion_run_id": run_id,
        "index_fingerprint": fingerprint,
        "progress_completed": "0",
        "progress_total": str(len(chunks)),
        "progress_unit": "chunks",
    }, token=lease_token)

    page_count = int(artifact["page_count"])
    chunk_count = len(chunks)
    done = 0
    try:
        for start in range(0, chunk_count, UPSERT_BATCH):
            if not lifecycle.renew_lease(job_id, lease_token, ttl_ms=LEASE_TTL_MS):
                raise IndexingError(
                    ERROR_LEASE_LOST,
                    "job lease lost during indexing; another worker may own the job",
                    retryable=True,
                )
            batch = chunks[start : start + UPSERT_BATCH]
            texts = [chunk["text"] for chunk in batch]
            try:
                encoded = embedder.embed(
                    EmbedRequest(texts=texts, input_type=InputType.DOCUMENT)
                )
            except LocalModelsError as exc:
                raise IndexingError(
                    ERROR_EMBEDDING_UNAVAILABLE,
                    f"embedding batch failed: {exc}",
                    retryable=True,
                ) from exc
            if ((encoded.model_id, encoded.revision, encoded.dim)
                    != (probe.model_id, probe.revision, probe.dim)):
                raise IndexingError(
                    ERROR_INDEX_FINGERPRINT_MISMATCH,
                    "embedding model identity changed during indexing", retryable=False,
                )
            dense = encoded.vectors
            if len(dense) != len(texts) or any(len(vector) != probe.dim for vector in dense):
                raise IndexingError(
                    ERROR_INDEX_VERIFY_FAILED, "invalid embedding batch shape", retryable=True,
                )
            sparse = bm25.embed_documents(texts)
            payloads = [
                _chunk_payload(
                    chunk, job=job, run_id=run_id, fingerprint=fingerprint,
                    page_count=page_count, chunk_count=chunk_count, created_at=created_at,
                )
                for chunk in batch
            ]
            lifecycle.set_progress(job_id, {"stage": "upsert"}, token=lease_token)
            q_retry("qdrant.upsert", functools.partial(
                index_repo.upsert_chunk_batch,
                doc_id=job["doc_id"], run_id=run_id, batch=payloads,
                dense_vectors=dense, sparse_vectors=sparse,
            ))
            done += len(batch)
            lifecycle.set_progress(job_id, {
                "stage": "embed", "progress_completed": str(done),
            }, token=lease_token)

        # verification before any publication marker
        lifecycle.set_progress(job_id, {"stage": "verify"}, token=lease_token)
        actual = q_retry("qdrant.count", lambda: index_repo.count_run(job["doc_id"], run_id))
        if actual != chunk_count:
            raise IndexingError(
                ERROR_INDEX_VERIFY_FAILED,
                f"exact count {actual} != expected {chunk_count} for run {run_id}",
                retryable=True, run_id=run_id,
            )
        q_retry("qdrant.head_complete",
                lambda: index_repo.mark_run_complete_on_head(job["doc_id"], run_id))
    except LeaseLostError:
        raise
    except IndexingError as exc:
        if exc.run_id is None:
            exc.run_id = run_id
        raise  # compensation is the CALLER's job (tasks.run_index saga step)
    except Exception as exc:  # qdrant/transport failures after bounded retries
        raise IndexingError(
            ERROR_QDRANT_UNAVAILABLE,
            f"indexing failed: {type(exc).__name__}: {exc}",
            retryable=True, run_id=run_id,
        ) from exc

    return IndexOutcome(run_id=run_id, fingerprint=fingerprint, chunk_count=chunk_count)
