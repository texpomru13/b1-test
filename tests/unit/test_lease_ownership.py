"""Job ownership (ADR 003 §5): the worker takes the lease BEFORE the job becomes
`running`, a heartbeat keeps it alive through parsing and indexing, and a worker
that lost its lease steps aside without touching the job state."""

import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from app.config.settings import Settings
from app.ingestion.errors import ParseError
from app.jobs import tasks
from app.jobs.indexer import IndexingError, IndexOutcome
from app.jobs.lease import LeaseKeeper


class FakeLifecycle:
    """In-memory stand-in for SyncJobLifecycle with the same guarded semantics."""

    def __init__(self, status: str = "queued") -> None:
        self.job: dict[str, str] = {"job_id": "job-1", "doc_id": "doc-1",
                                    "status": status, "attempt": "0",
                                    "content_type": "application/pdf"}
        self.lease: str | None = None
        self.events: list[str] = []
        self.publish_verdict = "OK"
        self.renewals = 0
        self.fail_renew_times = 0
        self._lock = threading.Lock()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return dict(self.job)

    def transition(self, job_id: str, *, expected: list[str], new_status: str,
                   extra: dict[str, str] | None = None) -> Any:
        previous = self.job["status"]
        ok = previous in expected
        if ok:
            self.job["status"] = new_status
            self.job.update(extra or {})
            self.events.append(f"status:{new_status}")
        return type("R", (), {"ok": ok, "verdict": "OK" if ok else "INVALID_STATE",
                              "previous_status": previous})()

    def set_fields(self, job_id: str, fields: dict[str, str]) -> None:
        self.job.update(fields)

    def incr_attempt(self, job_id: str) -> int:
        self.job["attempt"] = str(int(self.job["attempt"]) + 1)
        return int(self.job["attempt"])

    def acquire_lease(self, job_id: str, *, ttl_ms: int) -> str | None:
        with self._lock:
            if self.lease is not None:
                return None
            self.lease = uuid.uuid4().hex
            self.events.append("lease:acquired")
            return self.lease

    def renew_lease(self, job_id: str, token: str, *, ttl_ms: int) -> bool:
        with self._lock:
            self.renewals += 1
            if self.fail_renew_times > 0:
                self.fail_renew_times -= 1
                raise ConnectionError("redis hiccup")
            return self.lease == token

    def release_lease(self, job_id: str, token: str) -> None:
        with self._lock:
            if self.lease == token:
                self.lease = None
                self.events.append("lease:released")

    def publish(self, job_id: str, *, token: str, run_id: str) -> Any:
        verdict = self.publish_verdict if self.lease == token else "STALE_TOKEN"
        if verdict == "OK":
            self.job.update({"status": "succeeded", "accepted_run_id": run_id})
        return type("R", (), {"ok": verdict == "OK", "verdict": verdict,
                              "previous_status": self.job["status"]})()


class FakeIndexRepo:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def delete_run(self, doc_id: str, run_id: str) -> None:
        self.deleted.append((doc_id, run_id))


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, documents_dir=str(tmp_path))  # type: ignore[call-arg]


def _wait_until(predicate: Any, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ---------------- LeaseKeeper ----------------

def test_heartbeat_renews_until_stopped() -> None:
    lifecycle = FakeLifecycle()
    token = lifecycle.acquire_lease("job-1", ttl_ms=1000)
    assert token is not None
    keeper = LeaseKeeper(lifecycle, "job-1", token, ttl_ms=1000, interval_s=0.01)
    keeper.start()
    assert _wait_until(lambda: lifecycle.renewals >= 3)
    keeper.stop()
    settled = lifecycle.renewals
    time.sleep(0.05)
    assert lifecycle.renewals == settled  # no beats after stop
    assert keeper.lost is False


def test_heartbeat_reports_lost_lease() -> None:
    lifecycle = FakeLifecycle()
    token = lifecycle.acquire_lease("job-1", ttl_ms=1000)
    assert token is not None
    with LeaseKeeper(lifecycle, "job-1", token, ttl_ms=1000, interval_s=0.01) as keeper:
        lifecycle.lease = "someone-else"  # expired and taken over
        assert _wait_until(lambda: keeper.lost)


def test_heartbeat_survives_a_redis_hiccup() -> None:
    lifecycle = FakeLifecycle()
    token = lifecycle.acquire_lease("job-1", ttl_ms=1000)
    assert token is not None
    lifecycle.fail_renew_times = 2
    with LeaseKeeper(lifecycle, "job-1", token, ttl_ms=1000, interval_s=0.01) as keeper:
        assert _wait_until(lambda: lifecycle.renewals >= 4)
        assert keeper.lost is False  # errors are not a loss; it kept beating


# ---------------- run_index ownership ----------------

def test_lease_is_held_before_running_and_through_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle()
    seen: dict[str, Any] = {}

    def parse_spy(**kwargs: Any) -> Any:
        # what the reconciler would observe in the middle of a long parse
        seen["status"] = lifecycle.job["status"]
        seen["lease_held"] = lifecycle.lease is not None
        raise ParseError("stop here")

    monkeypatch.setattr(tasks, "parse_document", parse_spy)
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path))  # type: ignore[arg-type]
    assert outcome == "parse_failed:PARSE_FAILED"
    assert seen == {"status": "running", "lease_held": True}
    # ownership strictly brackets the state changes
    assert lifecycle.events == ["lease:acquired", "status:running", "status:failed",
                                "lease:released"]


def test_delivery_for_an_owned_job_changes_nothing(tmp_path: Path) -> None:
    lifecycle = FakeLifecycle(status="running")
    lifecycle.lease = "live-owner"
    before = dict(lifecycle.job)
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path))  # type: ignore[arg-type]
    assert outcome == "lease_unavailable"
    assert lifecycle.job == before and lifecycle.lease == "live-owner"


def _prepare_indexing(lifecycle: FakeLifecycle, monkeypatch: pytest.MonkeyPatch,
                      indexing: Any) -> None:
    result = type("ParseResultStub", (), {"page_count": 1, "chunks": [object()],
                                          "length_fn_id": "approx"})()
    monkeypatch.setattr(tasks, "parse_document", lambda **kwargs: result)
    monkeypatch.setattr(tasks, "save_parse_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(tasks, "_get_bm25", lambda settings: object())
    monkeypatch.setattr(tasks, "run_indexing", indexing)


def test_owner_publishes_under_its_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle()
    _prepare_indexing(lifecycle, monkeypatch, lambda **kwargs: IndexOutcome(
        run_id="run-1", fingerprint="fp", chunk_count=1))
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path),  # type: ignore[arg-type]
                              index_repo=FakeIndexRepo(), embedder=object())  # type: ignore[arg-type]
    assert outcome == "succeeded"
    assert lifecycle.job["status"] == "succeeded" and lifecycle.lease is None


def test_stale_owner_at_publish_never_touches_job_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle()

    def indexing(**kwargs: Any) -> IndexOutcome:
        lifecycle.lease = "new-owner"  # lease expired, another attempt owns the job
        return IndexOutcome(run_id="old-run", fingerprint="fp", chunk_count=1)

    _prepare_indexing(lifecycle, monkeypatch, indexing)
    repo = FakeIndexRepo()
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path),  # type: ignore[arg-type]
                              index_repo=repo, embedder=object())  # type: ignore[arg-type]
    assert outcome == "publish_rejected:STALE_TOKEN"
    assert lifecycle.job["status"] == "running"  # the new owner's state is intact
    assert "accepted_run_id" not in lifecycle.job
    assert lifecycle.lease == "new-owner"  # and its lease was not released


def test_lease_lost_mid_indexing_removes_only_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle()

    def indexing(**kwargs: Any) -> IndexOutcome:
        lifecycle.lease = "new-owner"
        raise IndexingError("LEASE_LOST", "lost", retryable=True, run_id="partial-run")

    _prepare_indexing(lifecycle, monkeypatch, indexing)
    repo = FakeIndexRepo()
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path),  # type: ignore[arg-type]
                              index_repo=repo, embedder=object())  # type: ignore[arg-type]
    assert outcome == "indexing_failed:LEASE_LOST"
    assert repo.deleted == [("doc-1", "partial-run")]
    assert lifecycle.job["status"] == "running"  # no compensating/failed clobbering
    assert lifecycle.lease == "new-owner"


def test_real_indexing_failure_still_compensates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = FakeLifecycle()

    def indexing(**kwargs: Any) -> IndexOutcome:
        raise IndexingError("QDRANT_UNAVAILABLE", "down", retryable=True, run_id="run-x")

    _prepare_indexing(lifecycle, monkeypatch, indexing)
    repo = FakeIndexRepo()
    outcome = tasks.run_index(lifecycle, "job-1", _settings(tmp_path),  # type: ignore[arg-type]
                              index_repo=repo, embedder=object())  # type: ignore[arg-type]
    assert outcome == "indexing_failed:QDRANT_UNAVAILABLE"
    assert repo.deleted == [("doc-1", "run-x")]
    assert lifecycle.job["status"] == "failed"
    assert lifecycle.job["error_code"] == "QDRANT_UNAVAILABLE"
