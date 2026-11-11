"""Outbox dispatcher: moves durable intents to Celery and runs the orphan janitor.

Crash contract (ADR 003 / plan §8.2): an intent is moved pending -> dispatching,
the Celery task is sent, the job is marked dispatched, then the intent is removed.
A crash between send and mark leaves the intent in `dispatching`; on restart it is
recovered back to `pending` and re-sent. Duplicate delivery is allowed — the
consumer deduplicates by job status, so no second logical job ever appears.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import redis

import app.jobs.tasks  # noqa: F401  - ensures task registration for send_task by name
from app.config.settings import Settings, get_settings
from app.domain.job import JobStatus
from app.jobs.celery_app import celery_app
from app.observability.logging import configure_logging, get_logger
from app.storage import keys
from app.storage.file_store import FileStore
from app.storage.job_lifecycle_sync import SyncJobLifecycle, create_sync_client
from app.storage.qdrant_repo import QdrantIndexRepository, create_sync_qdrant_client

logger = get_logger("dispatcher")

JANITOR_INTERVAL_S = 30.0
ORPHAN_GRACE_S = 60.0

Sender = Callable[[str], Any]


def default_sender(job_id: str) -> Any:
    return celery_app.send_task("app.jobs.index_document", args=[job_id])


def recover_stuck(client: redis.Redis) -> int:
    """Move intents stuck in `dispatching` (crash window) back to `pending`."""
    recovered = 0
    while True:
        moved = client.lmove(
            keys.outbox_dispatching_key(), keys.outbox_pending_key(), "LEFT", "RIGHT"
        )
        if moved is None:
            break
        recovered += 1
        logger.info("outbox_intent_recovered", job_id=str(moved))
    return recovered


def dispatch_one(
    client: redis.Redis,
    lifecycle: SyncJobLifecycle,
    job_id: str,
    *,
    sender: Sender,
) -> str:
    """Dispatch a single intent already moved to `dispatching`. Returns an outcome tag."""
    job = lifecycle.get_job(job_id)
    if job is None:
        client.lrem(keys.outbox_dispatching_key(), 1, job_id)
        logger.warning("outbox_intent_without_job", job_id=job_id)
        return "missing_job"

    already_dispatched = job.get("dispatched") == "1"
    if not already_dispatched:
        sender(job_id)  # may be re-sent after a crash window; consumer deduplicates
        result = lifecycle.transition(
            job_id, expected=[JobStatus.ACCEPTED.value], new_status=JobStatus.QUEUED.value
        )
        if not result.ok and result.verdict == "INVALID_STATE":
            logger.info("dispatch_state_already_advanced", job_id=job_id,
                        status=result.previous_status)
        lifecycle.mark_dispatched(job_id)
        outcome = "dispatched"
    else:
        outcome = "already_dispatched"
    client.lrem(keys.outbox_dispatching_key(), 1, job_id)
    return outcome


def run_janitor(client: redis.Redis, store: FileStore) -> list[str]:
    removed = store.cleanup_orphans(
        is_referenced=lambda doc_id: bool(client.exists(keys.doc_key(doc_id))),
        grace_s=ORPHAN_GRACE_S,
    )
    if removed:
        logger.info("orphan_files_removed", doc_ids=removed)
    return removed


def main(settings: Settings | None = None) -> None:  # pragma: no cover - process entry
    configure_logging()
    resolved = settings or get_settings()
    client = create_sync_client(resolved.redis_url, socket_timeout=10)
    lifecycle = SyncJobLifecycle(client)
    store = FileStore(
        Path(resolved.documents_dir), max_bytes=resolved.max_file_mb * 1024 * 1024
    )

    stop = {"flag": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("dispatcher_stopping", signal=signum)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    index_repo = QdrantIndexRepository(
        create_sync_qdrant_client(resolved.qdrant_url), resolved.qdrant_collection
    )
    recovered = recover_stuck(client)
    logger.info("dispatcher_started", recovered_intents=recovered)
    last_janitor = 0.0
    last_reconcile = 0.0

    while not stop["flag"]:
        try:
            try:
                moved = client.blmove(
                    keys.outbox_pending_key(), keys.outbox_dispatching_key(),
                    timeout=5, src="LEFT", dest="RIGHT",
                )
            except (redis.TimeoutError, TimeoutError):
                moved = None  # idle poll window elapsed — normal, not an error
            if moved is not None:
                dispatch_one(client, lifecycle, str(moved), sender=default_sender)
            if time.monotonic() - last_janitor > JANITOR_INTERVAL_S:
                run_janitor(client, store)
                last_janitor = time.monotonic()
            if time.monotonic() - last_reconcile > resolved.reconcile_interval_s:
                from app.jobs.reconciler import reconcile

                recover_stuck(client)  # intents stuck in `dispatching` (crash window)
                counters = reconcile(client, lifecycle, index_repo, store, resolved)
                if any(counters.values()):
                    logger.info("reconcile_pass", **counters)
                last_reconcile = time.monotonic()
        except redis.ConnectionError as exc:
            logger.warning("dispatcher_redis_unavailable", error=type(exc).__name__)
            time.sleep(2)
        except Exception as exc:
            logger.error("dispatcher_iteration_failed", error=type(exc).__name__)
            time.sleep(1)

    logger.info("dispatcher_stopped")


if __name__ == "__main__":  # pragma: no cover
    main()
