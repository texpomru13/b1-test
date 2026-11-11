"""Requeue/ownership acceptance against a REAL ephemeral Redis container (real Lua
scripts, real lease TTLs). Qdrant and local-models are not needed: indexing
itself is stubbed, the saga state machine around it is real.

Regression for the stuck-in-retry_wait defect: the reconciler used to requeue a
job whose worker was still parsing (running without a lease), the worker's
publication was then rejected, and the requeued intent was dropped by the
dispatcher because the job was already marked as dispatched.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import redis as redis_sync
from app.config.settings import Settings
from app.jobs import tasks
from app.jobs.dispatcher import dispatch_one
from app.jobs.indexer import IndexOutcome
from app.jobs.lease import LeaseKeeper
from app.jobs.reconciler import RETRY_WAIT_REDISPATCH_AFTER_S, reconcile
from app.storage import keys
from app.storage.file_store import FileStore
from app.storage.job_lifecycle_sync import SyncJobLifecycle
from app.storage.redis_repo import RedisJobRepository, create_redis_client

from tests.helpers.pdfgen import make_text_pdf

REDIS_IMAGE = (
    "redis:8.10.3@sha256:1f88291b4c57bdf249f02eb085a8669889b2d2954e91d4de5118a5da6b78dd53"
)
CONTAINER = "b1-test-redis-requeue"
PORT = 6414
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
def rc() -> Iterator[redis_sync.Redis]:
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


class NoQdrant:
    """The reconciler must not need Qdrant for any job in these scenarios."""

    def head_is_complete(self, doc_id: str, run_id: str) -> bool:
        return False

    def count_run(self, doc_id: str, run_id: str) -> int:
        return 0

    def count_other_runs(self, doc_id: str, accepted: str) -> int:
        return 0

    def delete_other_runs(self, doc_id: str, accepted: str) -> None:
        pass

    def delete_run(self, doc_id: str, run_id: str) -> None:
        pass


async def _seed(docs_root: Path) -> tuple[str, str]:
    repo = RedisJobRepository(create_redis_client(URL))
    job_id, doc_id = uuid.uuid4().hex, uuid.uuid4().hex
    result = await repo.submit(
        idem_key=f"k-{uuid.uuid4().hex[:8]}", fingerprint=uuid.uuid4().hex,
        job_id=job_id, doc_id=doc_id,
        job_fields={"title": "t", "file_sha256": "x",
                    "content_type": "application/pdf"},
    )
    await repo.aclose()
    assert result.verdict == "CREATED"
    doc_dir = docs_root / doc_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "original.pdf").write_bytes(
        make_text_pdf(["1. Subject\nGoods are delivered within ten working days."])
    )
    return job_id, doc_id


def _settings(docs_root: Path) -> Settings:
    return Settings(_env_file=None, documents_dir=str(docs_root), redis_url=URL)  # type: ignore[call-arg]


def _stub_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "_get_bm25", lambda settings: object())
    monkeypatch.setattr(tasks, "run_indexing", lambda **kwargs: IndexOutcome(
        run_id=uuid.uuid4().hex, fingerprint="fp", chunk_count=1))


def _dispatch_next(rc: redis_sync.Redis, lifecycle: SyncJobLifecycle, job_id: str,
                   sent: list[str]) -> str:
    """One dispatcher loop step for this job (drains unrelated intents)."""
    while True:
        moved = rc.lmove(keys.outbox_pending_key(), keys.outbox_dispatching_key(),
                         "LEFT", "RIGHT")
        assert moved is not None, "no intent for the job in the outbox"
        if moved == job_id:
            return dispatch_one(rc, lifecycle, job_id, sender=sent.append)
        rc.lrem(keys.outbox_dispatching_key(), 1, moved)


def _age(rc: redis_sync.Redis, job_id: str, seconds: float) -> None:
    stamp = (datetime.now(tz=UTC) - timedelta(seconds=seconds)).isoformat()
    rc.hset(keys.job_key(job_id), "updated_at", stamp)


async def test_reconciler_pass_during_parsing_does_not_requeue_a_live_worker(
    rc: redis_sync.Redis, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact production failure: a reconcile pass lands while the worker is
    still parsing. The job must stay `running` and be published normally."""
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)
    job_id, _doc = await _seed(tmp_path)
    _stub_indexing(monkeypatch)
    observed: dict[str, Any] = {}
    real_parse = tasks.parse_document

    def parse_with_reconcile_in_the_middle(**kwargs: Any) -> Any:
        observed["counters"] = reconcile(rc, lifecycle, NoQdrant(), store, settings)  # type: ignore[arg-type]
        observed["status"] = rc.hget(keys.job_key(job_id), "status")
        observed["lease"] = rc.exists(keys.lease_key(job_id))
        return real_parse(**kwargs)

    monkeypatch.setattr(tasks, "parse_document", parse_with_reconcile_in_the_middle)
    outcome = tasks.run_index(lifecycle, job_id, settings, index_repo=NoQdrant(),  # type: ignore[arg-type]
                              embedder=object())  # type: ignore[arg-type]

    assert observed["lease"] == 1  # the worker owned the job while parsing
    assert observed["counters"]["requeued"] == 0
    assert observed["status"] == "running"
    assert outcome == "succeeded"
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "succeeded" and job["accepted_run_id"]
    await asyncio.sleep(0.05)  # release = owner-only expiry in 1 ms
    assert rc.exists(keys.lease_key(job_id)) == 0  # released at the end


async def test_requeued_job_is_really_redelivered_and_completes(
    rc: redis_sync.Redis, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)
    job_id, _doc = await _seed(tmp_path)
    sent: list[str] = []

    # first delivery through the real dispatcher step
    assert _dispatch_next(rc, lifecycle, job_id, sent) == "dispatched"
    assert sent == [job_id] and rc.hget(keys.job_key(job_id), "dispatched") == "1"

    # the worker starts and dies: `running`, attempt counted, lease expired
    token = lifecycle.acquire_lease(job_id, ttl_ms=60_000)
    assert token is not None
    assert lifecycle.transition(job_id, expected=["queued"], new_status="running").ok
    lifecycle.incr_attempt(job_id)
    rc.delete(keys.lease_key(job_id))

    counters = reconcile(rc, lifecycle, NoQdrant(), store, settings)  # type: ignore[arg-type]
    assert counters["requeued"] == 1
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "retry_wait"
    assert job["dispatched"] == "0"  # cleared atomically with the requeue

    # the dispatcher must SEND it again (it used to drop it as already dispatched)
    assert _dispatch_next(rc, lifecycle, job_id, sent) == "dispatched"
    assert sent == [job_id, job_id]

    _stub_indexing(monkeypatch)
    assert tasks.run_index(lifecycle, job_id, settings, index_repo=NoQdrant(),  # type: ignore[arg-type]
                           embedder=object()) == "succeeded"  # type: ignore[arg-type]
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "succeeded" and job["attempt"] == "2"


async def test_stale_retry_wait_is_redispatched_only_when_nobody_owns_it(
    rc: redis_sync.Redis, tmp_path: Path
) -> None:
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)

    async def stuck_job() -> str:
        # the legacy stuck state: retry_wait, marked dispatched, no intent anywhere
        job_id, _ = await _seed(tmp_path)
        rc.lrem(keys.outbox_pending_key(), 0, job_id)
        lifecycle.transition(job_id, expected=["accepted"], new_status="retry_wait",
                             extra={"dispatched": "1", "attempt": "1"})
        return job_id

    def passes(job_id: str) -> int:
        before = rc.lrange(keys.outbox_pending_key(), 0, -1).count(job_id)
        reconcile(rc, lifecycle, NoQdrant(), store, settings)  # type: ignore[arg-type]
        return rc.lrange(keys.outbox_pending_key(), 0, -1).count(job_id) - before

    recent = await stuck_job()
    assert passes(recent) == 0  # may simply be waiting in the queue: left alone

    owned = await stuck_job()
    _age(rc, owned, RETRY_WAIT_REDISPATCH_AFTER_S + 60)
    assert lifecycle.acquire_lease(owned, ttl_ms=60_000) is not None
    assert passes(owned) == 0  # a worker is on it

    stale = await stuck_job()
    _age(rc, stale, RETRY_WAIT_REDISPATCH_AFTER_S + 60)
    assert passes(stale) == 1
    assert rc.hget(keys.job_key(stale), "dispatched") == "0"
    assert passes(stale) == 0  # one intent, no pile-up on the next pass
    sent: list[str] = []
    assert _dispatch_next(rc, lifecycle, stale, sent) == "dispatched" and sent == [stale]

    exhausted = await stuck_job()
    rc.hset(keys.job_key(exhausted), "attempt", str(settings.job_max_attempts))
    _age(rc, exhausted, RETRY_WAIT_REDISPATCH_AFTER_S + 60)
    assert passes(exhausted) == 0
    job = lifecycle.get_job(exhausted)
    assert job is not None and job["status"] == "failed"
    assert job["error_code"] == "ATTEMPTS_EXHAUSTED"  # bounded, never an endless loop


async def test_second_delivery_of_an_owned_job_is_a_noop(
    rc: redis_sync.Redis, tmp_path: Path
) -> None:
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    job_id, _doc = await _seed(tmp_path)
    owner_token = lifecycle.acquire_lease(job_id, ttl_ms=60_000)
    assert owner_token is not None
    lifecycle.transition(job_id, expected=["accepted"], new_status="running")
    before = lifecycle.get_job(job_id)

    assert tasks.run_index(lifecycle, job_id, settings) == "lease_unavailable"
    assert lifecycle.get_job(job_id) == before  # status, attempt, timestamps intact
    assert rc.get(keys.lease_key(job_id)) == owner_token  # the owner keeps its lease


async def test_heartbeat_outlives_the_ttl_and_expiry_follows_a_dead_owner(
    rc: redis_sync.Redis, tmp_path: Path
) -> None:
    lifecycle = SyncJobLifecycle(rc)
    job_id, _doc = await _seed(tmp_path)
    token = lifecycle.acquire_lease(job_id, ttl_ms=400)
    assert token is not None
    keeper = LeaseKeeper(lifecycle, job_id, token, ttl_ms=400, interval_s=0.1)
    keeper.start()
    await asyncio.sleep(1.2)  # three TTLs: only the heartbeat can keep the lease alive
    assert rc.get(keys.lease_key(job_id)) == token and keeper.lost is False
    keeper.stop()  # the owner dies: nothing renews the lease any more
    await asyncio.sleep(0.7)
    assert rc.exists(keys.lease_key(job_id)) == 0
