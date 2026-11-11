"""Celery tasks.

Stage 4: `index_document` = real parse (stage 3) + real indexing: dense vectors
via the local-models service, BM25 sparse vectors, batch upserts into the single
Qdrant collection, count verification, run_complete head marker and lease-gated
CAS publication (ADR 003). A new ingestion_run_id is created per attempt.

Ownership (ADR 003 §5): the worker takes the job lease BEFORE the job becomes
`running` and a heartbeat keeps it alive for the whole execution, parsing
included. `running` therefore always means "a live worker holds the lease" —
the reconciler relies on exactly that to tell a crashed worker from a busy one.
A worker that lost its lease steps aside and never touches the job state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.domain.job import (
    ERROR_ATTEMPTS_EXHAUSTED,
    TERMINAL_STATUSES,
    JobStage,
    JobStatus,
)
from app.inference.client import LocalModelsSyncClient
from app.ingestion.chunker import ApproxTokenLength, HFTokenizerLength, LengthFn
from app.ingestion.errors import ParseError
from app.ingestion.pipeline import parse_document, save_parse_artifacts
from app.jobs.celery_app import celery_app
from app.jobs.indexer import (
    ERROR_LEASE_LOST,
    ERROR_PUBLISH_FAILED,
    IndexingError,
    run_indexing,
)
from app.jobs.lease import LEASE_TTL_MS, LeaseKeeper
from app.observability.logging import get_logger
from app.retrieval.bm25 import Bm25Adapter
from app.storage.job_lifecycle_sync import SyncJobLifecycle, create_sync_client
from app.storage.qdrant_repo import QdrantIndexRepository, create_sync_qdrant_client

logger = get_logger("jobs.index")

_bm25_cache: dict[tuple[str, int], Bm25Adapter] = {}


def _get_bm25(settings: Settings) -> Bm25Adapter:
    key = (settings.bm25_language, settings.bm25_avg_len)
    if key not in _bm25_cache:
        _bm25_cache[key] = Bm25Adapter(
            language=settings.bm25_language, avg_len=float(settings.bm25_avg_len),
            cache_dir=settings.fastembed_cache_dir,
        )
    return _bm25_cache[key]


def _resolve_length_fn(settings: Settings) -> tuple[LengthFn, str]:
    if settings.embedding_tokenizer_path:
        path = Path(settings.embedding_tokenizer_path)
        exact = HFTokenizerLength(path)
        return exact, exact.id
    approx = ApproxTokenLength()
    return approx, approx.id


def _fail(
    lifecycle: SyncJobLifecycle, job_id: str, *, code: str, message: str, retryable: bool
) -> None:
    lifecycle.transition(
        job_id,
        expected=[JobStatus.RUNNING.value],
        new_status=JobStatus.FAILED.value,
        extra={
            "error_code": code,
            "error_message": message,
            "error_retryable": "1" if retryable else "0",
        },
    )


def run_index(
    lifecycle: SyncJobLifecycle,
    job_id: str,
    settings: Settings,
    *,
    index_repo: QdrantIndexRepository | None = None,
    embedder: LocalModelsSyncClient | None = None,
    bm25: Bm25Adapter | None = None,
) -> str:
    """Task body, separated for direct testing against real services."""
    job = lifecycle.get_job(job_id)
    if job is None:
        logger.error("job_not_found", job_id=job_id)
        return "missing"
    if JobStatus(job["status"]) in TERMINAL_STATUSES:
        logger.info("duplicate_delivery_short_circuit", job_id=job_id, status=job["status"])
        return f"duplicate:{job['status']}"

    # Ownership first: without the lease this delivery must not change anything.
    lease_token = lifecycle.acquire_lease(job_id, ttl_ms=LEASE_TTL_MS)
    if lease_token is None:
        # Another delivery of the same job is executing right now (or its lease
        # has not expired yet). The owner — or the reconciler after the lease
        # expires — is responsible for the job; this delivery is a no-op.
        logger.info("delivery_skipped_job_owned", job_id=job_id, status=job["status"])
        return "lease_unavailable"
    keeper = LeaseKeeper(lifecycle, job_id, lease_token)
    keeper.start()
    try:
        return _run_owned(
            lifecycle, job_id, job, settings, lease_token=lease_token, keeper=keeper,
            index_repo=index_repo, embedder=embedder, bm25=bm25,
        )
    finally:
        keeper.stop()
        lifecycle.release_lease(job_id, lease_token)


def _run_owned(
    lifecycle: SyncJobLifecycle,
    job_id: str,
    job: dict[str, Any],
    settings: Settings,
    *,
    lease_token: str,
    keeper: LeaseKeeper,
    index_repo: QdrantIndexRepository | None,
    embedder: LocalModelsSyncClient | None,
    bm25: Bm25Adapter | None,
) -> str:
    """The task body executed while holding the job lease."""
    started = lifecycle.transition(
        job_id,
        expected=[JobStatus.QUEUED.value, JobStatus.ACCEPTED.value, JobStatus.RETRY_WAIT.value],
        new_status=JobStatus.RUNNING.value,
        extra={"stage": JobStage.PARSE.value},
    )
    if not started.ok:
        logger.warning("start_transition_rejected", job_id=job_id,
                       verdict=started.verdict, previous=started.previous_status)
        return f"rejected:{started.verdict}"

    attempt = lifecycle.incr_attempt(job_id)
    if attempt > settings.job_max_attempts:
        _fail(lifecycle, job_id, code=ERROR_ATTEMPTS_EXHAUSTED,
              message=f"exceeded {settings.job_max_attempts} execution attempts",
              retryable=False)
        return "attempts_exhausted"

    doc_dir = Path(settings.documents_dir) / job["doc_id"]
    length_fn, length_fn_id = _resolve_length_fn(settings)
    try:
        result = parse_document(
            doc_dir=doc_dir,
            content_type=job.get("content_type", ""),
            max_pages=settings.max_pages,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            length_fn=length_fn,
            length_fn_id=length_fn_id,
            progress=lambda stage: lifecycle.set_fields(job_id, {"stage": stage}),
        )
    except ParseError as exc:
        logger.warning("parse_failed", job_id=job_id, code=exc.code,
                       detail=exc.detail or "")
        _fail(lifecycle, job_id, code=exc.code, message=exc.message,
              retryable=exc.retryable)
        return f"parse_failed:{exc.code}"

    save_parse_artifacts(
        doc_dir, result,
        chunk_tokens=settings.chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens,
    )
    lifecycle.set_fields(job_id, {
        "page_count": str(result.page_count),
        "chunk_count": str(len(result.chunks)),
        "length_fn": result.length_fn_id,
    })
    logger.info("parse_completed", job_id=job_id, pages=result.page_count,
                chunks=len(result.chunks))

    if keeper.lost:
        # The lease expired while parsing (e.g. Redis was unreachable longer than
        # the TTL): the job may already belong to another attempt. Step aside.
        logger.warning("stale_owner_steps_aside", job_id=job_id, phase="parse")
        return "lease_lost"

    # --- stage 4: real indexing under the lease, published via CAS ---
    repo = index_repo or QdrantIndexRepository(
        create_sync_qdrant_client(settings.qdrant_url), settings.qdrant_collection
    )
    try:
        outcome = run_indexing(
            job=_refresh(lifecycle, job_id, job),
            job_id=job_id,
            lifecycle=lifecycle,
            lease_token=lease_token,
            index_repo=repo,
            embedder=embedder or LocalModelsSyncClient(settings.local_models_url),
            bm25=bm25 or _get_bm25(settings),
            settings=settings,
        )
        published = lifecycle.publish(job_id, token=lease_token, run_id=outcome.run_id)
        if not published.ok:
            if published.verdict == "STALE_TOKEN":
                # Not the owner any more: never touch the job state. The verified
                # run stays in Qdrant, invisible to QA: the reconciler publishes it
                # (forward recovery) or removes it once another run is accepted.
                logger.warning("stale_owner_steps_aside", job_id=job_id,
                               phase="publish", run_id=outcome.run_id)
                return "publish_rejected:STALE_TOKEN"
            logger.warning("publish_rejected", job_id=job_id,
                           verdict=published.verdict, status=published.previous_status)
            _fail(lifecycle, job_id, code=ERROR_PUBLISH_FAILED,
                  message=f"publication CAS rejected: {published.verdict} "
                          f"({published.previous_status})",
                  retryable=False)
            return f"publish_rejected:{published.verdict}"
        logger.info("indexing_published", job_id=job_id, run_id=outcome.run_id,
                    chunks=outcome.chunk_count)
        return "succeeded"
    except IndexingError as exc:
        logger.warning("indexing_failed", job_id=job_id, code=exc.code)
        if exc.code == ERROR_LEASE_LOST:
            # Stale owner: remove only our own partial run and leave the job
            # state to whoever owns the job now (or to the reconciler).
            _remove_own_run(repo, job["doc_id"], exc.run_id)
            return f"indexing_failed:{exc.code}"
        _compensate_and_fail(
            lifecycle, job_id, index_repo=repo, doc_id=job["doc_id"], error=exc,
        )
        return f"indexing_failed:{exc.code}"


def _remove_own_run(
    index_repo: QdrantIndexRepository, doc_id: str, run_id: str | None
) -> None:
    if not run_id:
        return
    try:
        index_repo.delete_run(doc_id, run_id)  # doc+run filter only
        logger.info("incomplete_run_removed", doc_id=doc_id, run_id=run_id)
    except Exception as exc:
        # Left for the reconciler's stale-run cleanup; never visible to QA.
        logger.warning("own_run_cleanup_failed", doc_id=doc_id, run_id=run_id,
                       error=type(exc).__name__)


def _compensate_and_fail(
    lifecycle: SyncJobLifecycle,
    job_id: str,
    *,
    index_repo: QdrantIndexRepository,
    doc_id: str,
    error: IndexingError,
) -> None:
    """Saga compensation (§8.4): running -> compensating -> (failed |
    cleanup_pending). The incomplete run is removed with a doc+run filter ONLY;
    when Qdrant is unavailable the job becomes cleanup_pending and the
    reconciler retries later — it is never shown as succeeded or silently ready.
    """
    error_fields = {
        "error_code": error.code,
        "error_message": error.message,
        "error_retryable": "1" if error.retryable else "0",
    }
    lifecycle.transition(
        job_id,
        expected=[JobStatus.RUNNING.value, JobStatus.VALIDATING.value],
        new_status=JobStatus.COMPENSATING.value,
        extra=error_fields,
    )
    cleanup_ok = True
    if error.run_id:
        try:
            index_repo.delete_run(doc_id, error.run_id)
            logger.info("incomplete_run_removed", doc_id=doc_id, run_id=error.run_id)
        except Exception as cleanup_exc:
            cleanup_ok = False
            logger.warning("compensation_unavailable", doc_id=doc_id,
                           run_id=error.run_id, error=type(cleanup_exc).__name__)
    lifecycle.transition(
        job_id,
        expected=[JobStatus.COMPENSATING.value],
        new_status=(
            JobStatus.FAILED.value if cleanup_ok else JobStatus.CLEANUP_PENDING.value
        ),
        extra=error_fields,
    )


def _refresh(lifecycle: SyncJobLifecycle, job_id: str, fallback: dict[str, Any]) -> dict[str, Any]:
    return lifecycle.get_job(job_id) or fallback


@celery_app.task(name="app.jobs.index_document")
def index_document(job_id: str) -> str:
    settings = get_settings()
    lifecycle = SyncJobLifecycle(create_sync_client(settings.redis_url))
    return run_index(lifecycle, job_id, settings)


@celery_app.task(name="app.jobs.ping")
def ping() -> str:
    return "pong"
