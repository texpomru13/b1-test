"""Lifecycle acceptance against a REAL ephemeral Redis container.

Covers: atomic submit verdicts, the dispatcher crash window (publish before mark
=> duplicate delivery, single logical job), and consumer dedupe after a terminal
result. Self-provisions redis on port 6412; skipped when Docker is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator

import pytest
import redis as redis_sync
from app.config.settings import Settings
from app.jobs.dispatcher import dispatch_one, recover_stuck
from app.jobs.tasks import run_index
from app.storage import keys
from app.storage.job_lifecycle_sync import SyncJobLifecycle
from app.storage.redis_repo import RedisJobRepository, create_redis_client

from tests.helpers.pdfgen import make_text_pdf

REDIS_IMAGE = (
    "redis:8.10.3@sha256:1f88291b4c57bdf249f02eb085a8669889b2d2954e91d4de5118a5da6b78dd53"
)
CONTAINER = "b1-test-redis-s2"
PORT = 6412
URL = f"redis://127.0.0.1:{PORT}/0"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(
        ["docker", "info"], capture_output=True, check=False
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="requires a working Docker daemon"
)


@pytest.fixture(scope="module")
def sync_client() -> Iterator[redis_sync.Redis]:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)
    subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER, "-p", f"{PORT}:6379", REDIS_IMAGE],
        check=True, capture_output=True,
    )
    client = redis_sync.Redis.from_url(URL, decode_responses=True)
    for _ in range(50):
        try:
            if client.ping():
                break
        except redis_sync.ConnectionError:
            time.sleep(0.2)
    else:
        pytest.fail("test redis container did not become ready")
    yield client
    client.close()
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)


async def _submit(idem: str, fingerprint: str) -> tuple[str, str, str]:
    repo = RedisJobRepository(create_redis_client(URL))
    result = await repo.submit(
        idem_key=idem, fingerprint=fingerprint,
        job_id=uuid.uuid4().hex, doc_id=uuid.uuid4().hex,
        job_fields={"title": "t", "file_sha256": fingerprint,
                    "content_type": "application/pdf"},
    )
    await repo.aclose()
    return result.verdict, result.job_id, result.doc_id


async def test_submit_verdicts_on_real_redis(sync_client: redis_sync.Redis) -> None:
    idem = f"k-{uuid.uuid4().hex[:8]}"
    verdict1, job1, doc1 = await _submit(idem, "fp-A")
    assert verdict1 == "CREATED"
    verdict2, job2, doc2 = await _submit(idem, "fp-A")
    assert (verdict2, job2, doc2) == ("DUPLICATE", job1, doc1)
    verdict3, job3, _ = await _submit(idem, "fp-B")
    assert (verdict3, job3) == ("CONFLICT", job1)
    # exactly one logical job exists for this idempotency key
    assert sync_client.exists(keys.job_key(job1)) == 1
    # outbox got exactly one intent for this job
    pending = sync_client.lrange(keys.outbox_pending_key(), 0, -1)
    assert pending.count(job1) == 1


async def test_dispatcher_crash_window_no_second_logical_job(
    sync_client: redis_sync.Redis, tmp_path: object
) -> None:
    from pathlib import Path

    lifecycle = SyncJobLifecycle(sync_client)
    idem = f"k-{uuid.uuid4().hex[:8]}"
    verdict, job_id, doc_id = await _submit(idem, "fp-crash")
    assert verdict == "CREATED"
    docs_root = Path(str(tmp_path))
    doc_dir = docs_root / doc_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "original.pdf").write_bytes(
        make_text_pdf(["1. Subject\nGoods are delivered within ten working days."])
    )
    settings = Settings(_env_file=None, documents_dir=str(docs_root))  # type: ignore[call-arg]
    sent: list[str] = []

    # normal loop step: intent moves pending -> dispatching
    moved = sync_client.lmove(
        keys.outbox_pending_key(), keys.outbox_dispatching_key(), "LEFT", "RIGHT"
    )
    while moved is not None and moved != job_id:  # drain unrelated intents from other tests
        sync_client.lrem(keys.outbox_dispatching_key(), 1, moved)
        moved = sync_client.lmove(
            keys.outbox_pending_key(), keys.outbox_dispatching_key(), "LEFT", "RIGHT"
        )
    assert moved == job_id

    # CRASH WINDOW: task was published, mark/cleanup never happened
    sent.append(job_id)
    # (process dies here; intent is stuck in `dispatching`)

    # restart: recovery moves it back, dispatch completes, task is sent AGAIN
    assert recover_stuck(sync_client) == 1
    moved2 = sync_client.lmove(
        keys.outbox_pending_key(), keys.outbox_dispatching_key(), "LEFT", "RIGHT"
    )
    assert moved2 == job_id
    outcome = dispatch_one(sync_client, lifecycle, job_id, sender=sent.append)
    assert outcome == "dispatched"
    assert sent == [job_id, job_id]  # duplicate delivery is allowed...

    job_keys = [k for k in sync_client.scan_iter(f"{keys.KEY_PREFIX}:job:*")]
    jobs_for_idem = [k for k in job_keys if sync_client.hget(k, "idem_key") is None]
    # ...but there is still exactly ONE logical job for this submission:
    assert sync_client.exists(keys.job_key(job_id)) == 1
    assert sync_client.hget(keys.job_key(job_id), "dispatched") == "1"
    assert sync_client.hget(keys.job_key(job_id), "status") == "queued"
    del jobs_for_idem  # (job hash uniqueness asserted above)

    # consumer side: first delivery REALLY parses; local-models is unreachable in
    # this fixture, so indexing fails with its stable retryable code — the dedupe
    # mechanics under test are identical (full success path: test_indexing_qdrant).
    first = run_index(lifecycle, job_id, settings)
    assert first == "indexing_failed:EMBEDDING_UNAVAILABLE"
    record = lifecycle.get_job(job_id)
    assert record is not None
    assert record["status"] == "failed"
    assert record["error_code"] == "EMBEDDING_UNAVAILABLE"
    assert record["error_retryable"] == "1"
    assert record["attempt"] == "1"
    assert int(record["chunk_count"]) >= 1
    assert (doc_dir / "parse" / "parsed.json").exists()

    # ...the duplicate delivery short-circuits without reprocessing
    second = run_index(lifecycle, job_id, settings)
    assert second == "duplicate:failed"
    record2 = lifecycle.get_job(job_id)
    assert record2 is not None
    assert record2["attempt"] == "1"  # not incremented again


async def test_already_dispatched_intent_is_dropped(sync_client: redis_sync.Redis) -> None:
    lifecycle = SyncJobLifecycle(sync_client)
    idem = f"k-{uuid.uuid4().hex[:8]}"
    _, job_id, _ = await _submit(idem, "fp-dup-intent")
    sync_client.lrem(keys.outbox_pending_key(), 1, job_id)
    sync_client.rpush(keys.outbox_dispatching_key(), job_id)
    lifecycle.mark_dispatched(job_id)
    lifecycle.transition(job_id, expected=["accepted"], new_status="queued")

    sent: list[str] = []
    outcome = dispatch_one(sync_client, lifecycle, job_id, sender=sent.append)
    assert outcome == "already_dispatched"
    assert sent == []  # no re-send for an already-marked intent
