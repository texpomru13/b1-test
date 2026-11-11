"""Indexing stage: parse artifact -> embedded, BM25-encoded, verified, published run.

Flow per ADR 003: new ingestion_run_id per attempt; lease renewed on every batch;
count verification; run_complete marker on the REAL head chunk (index 0); then
lease-gated Redis CAS publication. Partial failures never publish; the incomplete
run is removed best-effort with a (doc_id AND run_id) filter only.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.domain.index_fingerprint import compute_index_fingerprint
from app.inference.client import LocalModelsError, LocalModelsSyncClient
from app.inference.contract import EmbedRequest, InputType
from app.jobs.lease import LEASE_TTL_MS
from app.observability.logging import get_logger
from app.reliability.policy import retry_sync
from app.retrieval.bm25 import Bm25Adapter
from app.storage.job_lifecycle_sync import SyncJobLifecycle
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
        "doc_title": job.get("title") or None,
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

    fingerprint = compute_index_fingerprint(
        embedding_model_id=probe.model_id,
        embedding_revision=probe.revision,
        embedding_dim=probe.dim,
        instruction_version=settings.embedding_instruction_version,
        chunk_tokens=artifact["chunk_tokens"],
        overlap_tokens=artifact["overlap_tokens"],
        length_fn_id=artifact["length_fn"],
        bm25_language=settings.bm25_language,
        bm25_avg_len=settings.bm25_avg_len,
        fastembed_version=importlib.metadata.version("fastembed"),
    )

    def q_retry[R](op_name: str, fn: Callable[[], R]) -> R:
        # Bounded retry on a temporarily unavailable vector store (§8.6); schema
        # and fingerprint mismatches are not transient and are never retried.
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

    run_id = uuid.uuid4().hex  # NEW run per attempt: old runs never mix (ADR 003)
    created_at = utc_now_iso()
    lifecycle.set_fields(job_id, {
        "stage": "embed",
        "ingestion_run_id": run_id,
        "index_fingerprint": fingerprint,
        "progress_completed": "0",
        "progress_total": str(len(chunks)),
        "progress_unit": "chunks",
    })

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
                dense = embedder.embed(
                    EmbedRequest(texts=texts, input_type=InputType.DOCUMENT)
                ).vectors
            except LocalModelsError as exc:
                raise IndexingError(
                    ERROR_EMBEDDING_UNAVAILABLE,
                    f"embedding batch failed: {exc}",
                    retryable=True,
                ) from exc
            sparse = bm25.embed_documents(texts)
            payloads = [
                _chunk_payload(
                    chunk, job=job, run_id=run_id, fingerprint=fingerprint,
                    page_count=page_count, chunk_count=chunk_count, created_at=created_at,
                )
                for chunk in batch
            ]
            lifecycle.set_fields(job_id, {"stage": "upsert"})
            q_retry("qdrant.upsert", functools.partial(
                index_repo.upsert_chunk_batch,
                doc_id=job["doc_id"], run_id=run_id, batch=payloads,
                dense_vectors=dense, sparse_vectors=sparse,
            ))
            done += len(batch)
            lifecycle.set_fields(job_id, {
                "stage": "embed", "progress_completed": str(done),
            })

        # verification before any publication marker
        lifecycle.set_fields(job_id, {"stage": "verify"})
        actual = q_retry("qdrant.count", lambda: index_repo.count_run(job["doc_id"], run_id))
        if actual != chunk_count:
            raise IndexingError(
                ERROR_INDEX_VERIFY_FAILED,
                f"exact count {actual} != expected {chunk_count} for run {run_id}",
                retryable=True, run_id=run_id,
            )
        q_retry("qdrant.head_complete",
                lambda: index_repo.mark_run_complete_on_head(job["doc_id"], run_id))
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


def _cleanup_incomplete_run(
    index_repo: QdrantIndexRepository, doc_id: str, run_id: str
) -> None:
    """Best-effort removal of the UNPUBLISHED run only (doc_id AND run_id)."""
    try:
        index_repo.delete_run(doc_id, run_id)
        logger.info("incomplete_run_removed", doc_id=doc_id, run_id=run_id)
    except Exception as exc:
        logger.warning("incomplete_run_cleanup_failed", doc_id=doc_id, run_id=run_id,
                       error=type(exc).__name__)
