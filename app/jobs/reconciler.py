"""Reconciler (ADR 003 / plan §8): repairs every crash window the saga leaves.

- Jobs stuck in running/validating with an EXPIRED lease (a live worker always
  holds the lease — it is taken before `running` and kept by a heartbeat):
  * forward recovery when the run is actually complete (count matches + head
    marked): re-acquire the lease and finish the publication CAS;
  * otherwise -> retry_wait + a fresh outbox intent with the `dispatched` flag
    cleared, so the dispatcher really re-sends the job (the next attempt creates
    a NEW ingestion_run_id; bounded by JOB_MAX_ATTEMPTS counted in Redis).
- Jobs sitting in retry_wait with no owner and no pending intent for longer
  than RETRY_WAIT_REDISPATCH_AFTER_S: re-dispatched (safety net against a lost
  delivery; duplicates are harmless — the consumer dedupes by lease and status).
- cleanup_pending jobs: retry the run-scoped compensation, then finish as failed.
- succeeded jobs: delete leftover points of NON-accepted runs (doc-scoped
  must_not accepted_run — never another doc, never the accepted run).
- failed jobs older than FAILED_SOURCE_TTL_S: remove source files (job record
  and its terminal status remain).
"""

from __future__ import annotations

from typing import Any

import redis

from app.config.settings import Settings
from app.domain.job import ERROR_ATTEMPTS_EXHAUSTED, JobStatus
from app.observability.logging import get_logger
from app.storage import keys
from app.storage.file_store import FileStore
from app.storage.job_lifecycle_sync import SyncJobLifecycle
from app.storage.qdrant_repo import QdrantIndexRepository
from app.storage.redis_repo import utc_now_iso

logger = get_logger("reconciler")

# A retry_wait job normally leaves that state within seconds of being re-sent.
# Past this age with nobody owning it and nothing queued for it, the delivery is
# considered lost and the job is re-dispatched.
RETRY_WAIT_REDISPATCH_AFTER_S = 300.0


def _lease_alive(client: redis.Redis, job_id: str) -> bool:
    return bool(client.exists(keys.lease_key(job_id)))


def _requeue(client: redis.Redis, lifecycle: SyncJobLifecycle, job: dict[str, Any]) -> str:
    job_id = job["job_id"]
    attempts = int(job.get("attempt", "0"))
    max_attempts = int(job.get("_max_attempts", "3"))
    if attempts >= max_attempts:
        lifecycle.transition(
            job_id,
            expected=[JobStatus.RUNNING.value, JobStatus.VALIDATING.value,
                      JobStatus.RETRY_WAIT.value],
            new_status=JobStatus.FAILED.value,
            extra={"error_code": ERROR_ATTEMPTS_EXHAUSTED,
                   "error_message": f"exceeded {max_attempts} execution attempts "
                                    "(reconciler)",
                   "error_retryable": "0"},
        )
        return "attempts_exhausted"
    moved = lifecycle.transition(
        job_id,
        expected=[JobStatus.RUNNING.value, JobStatus.VALIDATING.value],
        new_status=JobStatus.RETRY_WAIT.value,
        # The dispatcher drops intents of jobs it already sent; clearing the flag
        # in the same atomic step makes the requeue an actual re-delivery.
        extra={"dispatched": "0"},
    )
    if moved.ok:
        intent_added = _push_intent_once(client, job_id)
        logger.info("job_requeued_after_crash", job_id=job_id, attempt=attempts,
                    intent_added=intent_added)
        return "requeued"
    return f"requeue_skipped:{moved.verdict}"


def _intent_pending(client: redis.Redis, job_id: str) -> bool:
    return (
        client.lpos(keys.outbox_pending_key(), job_id) is not None
        or client.lpos(keys.outbox_dispatching_key(), job_id) is not None
    )


def _push_intent_once(client: redis.Redis, job_id: str) -> bool:
    # Avoid piling up duplicate intents (consumer dedupe stays the real
    # guarantee; this just keeps the outbox tidy).
    if _intent_pending(client, job_id):
        return False
    client.rpush(keys.outbox_pending_key(), job_id)
    return True


def _redispatch_stale_retry_wait(
    client: redis.Redis, lifecycle: SyncJobLifecycle, job: dict[str, Any],
    now_epoch: float,
) -> str:
    """retry_wait with no owner, nothing queued and no progress for a while:
    the re-delivery was lost (or never sent) — send it again."""
    job_id = job["job_id"]
    if _lease_alive(client, job_id) or _intent_pending(client, job_id):
        return "in_flight"
    updated = _epoch_from_iso(job.get("updated_at", ""))
    if updated and now_epoch - updated < RETRY_WAIT_REDISPATCH_AFTER_S:
        return "recent"
    attempts = int(job.get("attempt", "0"))
    max_attempts = int(job.get("_max_attempts", "3"))
    if attempts >= max_attempts:
        lifecycle.transition(
            job_id,
            expected=[JobStatus.RETRY_WAIT.value],
            new_status=JobStatus.FAILED.value,
            extra={"error_code": ERROR_ATTEMPTS_EXHAUSTED,
                   "error_message": f"exceeded {max_attempts} execution attempts "
                                    "(reconciler)",
                   "error_retryable": "0"},
        )
        return "attempts_exhausted"
    touched = lifecycle.transition(
        job_id,
        expected=[JobStatus.RETRY_WAIT.value],
        new_status=JobStatus.RETRY_WAIT.value,  # refreshes updated_at
        extra={"dispatched": "0"},
    )
    if not touched.ok:
        return f"redispatch_skipped:{touched.verdict}"
    _push_intent_once(client, job_id)
    logger.info("stale_retry_wait_redispatched", job_id=job_id, attempt=attempts)
    return "redispatched"


def _forward_recover(
    client: redis.Redis,
    lifecycle: SyncJobLifecycle,
    index_repo: QdrantIndexRepository,
    job: dict[str, Any],
) -> bool:
    """Publish a run that was fully verified before the crash (head complete)."""
    job_id, doc_id = job["job_id"], job["doc_id"]
    run_id = job.get("ingestion_run_id")
    chunk_count = job.get("chunk_count")
    if not run_id or not chunk_count:
        return False
    try:
        if not index_repo.head_is_complete(doc_id, run_id):
            return False
        if index_repo.count_run(doc_id, run_id) != int(chunk_count):
            return False
    except Exception as exc:
        logger.warning("forward_recovery_check_failed", job_id=job_id,
                       error=type(exc).__name__)
        return False
    token = lifecycle.acquire_lease(job_id, ttl_ms=30_000)
    if token is None:
        return False  # someone owns it; let them finish
    try:
        published = lifecycle.publish(job_id, token=token, run_id=run_id)
        if published.ok:
            logger.info("forward_recovery_published", job_id=job_id, run_id=run_id)
            return True
        return False
    finally:
        lifecycle.release_lease(job_id, token)


def _finish_cleanup_pending(
    lifecycle: SyncJobLifecycle, index_repo: QdrantIndexRepository, job: dict[str, Any]
) -> str:
    job_id, doc_id = job["job_id"], job["doc_id"]
    run_id = job.get("ingestion_run_id", "")
    try:
        if run_id:
            index_repo.delete_run(doc_id, run_id)
    except Exception as exc:
        logger.warning("cleanup_retry_failed", job_id=job_id, error=type(exc).__name__)
        return "cleanup_still_pending"
    lifecycle.transition(
        job_id,
        expected=[JobStatus.CLEANUP_PENDING.value],
        new_status=JobStatus.FAILED.value,
    )
    logger.info("cleanup_completed", job_id=job_id, run_id=run_id)
    return "cleaned"


def _epoch_from_iso(value: str) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def reconcile(
    client: redis.Redis,
    lifecycle: SyncJobLifecycle,
    index_repo: QdrantIndexRepository,
    store: FileStore,
    settings: Settings,
) -> dict[str, int]:
    """One reconciliation pass; returns action counters (used by tests/logs)."""
    counters = {"forward_recovered": 0, "requeued": 0, "redispatched": 0,
                "cleaned": 0, "stale_runs_removed": 0, "expired_sources_removed": 0}
    now_epoch = _epoch_from_iso(utc_now_iso())
    for key in client.scan_iter(f"{keys.KEY_PREFIX}:job:*"):
        raw = client.hgetall(str(key))
        if not raw:
            continue
        job = {str(k): str(v) for k, v in raw.items()}
        job["_max_attempts"] = str(settings.job_max_attempts)
        status = job.get("status", "")
        job_id = job["job_id"]

        if status in (JobStatus.RUNNING.value, JobStatus.VALIDATING.value):
            if _lease_alive(client, job_id):
                continue  # a live worker owns it
            if _forward_recover(client, lifecycle, index_repo, job):
                counters["forward_recovered"] += 1
            else:
                outcome = _requeue(client, lifecycle, job)
                if outcome == "requeued":
                    counters["requeued"] += 1

        elif status == JobStatus.RETRY_WAIT.value:
            outcome = _redispatch_stale_retry_wait(client, lifecycle, job, now_epoch)
            if outcome == "redispatched":
                counters["redispatched"] += 1

        elif status == JobStatus.CLEANUP_PENDING.value:
            if _finish_cleanup_pending(lifecycle, index_repo, job) == "cleaned":
                counters["cleaned"] += 1

        elif status == JobStatus.SUCCEEDED.value:
            accepted = job.get("accepted_run_id", "")
            if accepted:
                try:
                    if index_repo.count_other_runs(job["doc_id"], accepted) > 0:
                        index_repo.delete_other_runs(job["doc_id"], accepted)
                        counters["stale_runs_removed"] += 1
                except Exception as exc:
                    logger.warning("stale_run_cleanup_failed", job_id=job_id,
                                   error=type(exc).__name__)

        elif status == JobStatus.FAILED.value:
            updated = _epoch_from_iso(job.get("updated_at", ""))
            if updated and now_epoch - updated > settings.failed_source_ttl_s:
                doc_dir = store.doc_dir(job["doc_id"])
                if doc_dir.exists():
                    store.delete_doc_files(job["doc_id"])
                    counters["expired_sources_removed"] += 1
                    logger.info("failed_source_expired", job_id=job_id,
                                doc_id=job["doc_id"])
    return counters
