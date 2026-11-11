"""Stage-7 fault-injection acceptance on REAL Redis + Qdrant + local-models.

Each scenario simulates a specific crash window of the indexing saga and then
proves the documented recovery invariant (ADR 003).
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import redis as redis_sync
from app.config.settings import Settings
from app.inference.client import LocalModelsSyncClient
from app.jobs.indexer import run_indexing
from app.jobs.reconciler import reconcile
from app.jobs.tasks import _get_bm25, run_index
from app.storage import keys
from app.storage.file_store import FileStore
from app.storage.job_lifecycle_sync import SyncJobLifecycle
from app.storage.qdrant_repo import (
    QdrantIndexRepository,
    create_sync_qdrant_client,
)
from app.storage.redis_repo import RedisJobRepository, create_redis_client
from qdrant_client import QdrantClient

from tests.helpers.pdfgen import make_text_pdf

REDIS_IMAGE = (
    "redis:8.10.3@sha256:1f88291b4c57bdf249f02eb085a8669889b2d2954e91d4de5118a5da6b78dd53"
)
QDRANT_IMAGE = (
    "qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
)
REDIS_PORT, QDRANT_PORT, LM_PORT = 6417, 6418, 18084
REDIS_URL = f"redis://127.0.0.1:{REDIS_PORT}/0"
QDRANT_URL = f"http://127.0.0.1:{QDRANT_PORT}"
LM_URL = f"http://127.0.0.1:{LM_PORT}"
LM_DIR = Path(__file__).resolve().parents[2] / "services" / "local-models"


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True, check=False
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not (_docker_ok() and (LM_DIR / ".venv").exists()),
    reason="requires Docker and the local-models venv",
)


@pytest.fixture(scope="module")
def services() -> Iterator[dict[str, Any]]:
    for name in ("b1-test-redis-s7", "b1-test-qdrant-s7"):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(["docker", "run", "-d", "--name", "b1-test-redis-s7",
                    "-p", f"{REDIS_PORT}:6379", REDIS_IMAGE],
                   check=True, capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", "b1-test-qdrant-s7",
                    "-p", f"{QDRANT_PORT}:6333", QDRANT_IMAGE],
                   check=True, capture_output=True)
    import os

    lm = subprocess.Popen(
        [str(LM_DIR / ".venv/bin/uvicorn"), "app_lm.server:app",
         "--host", "127.0.0.1", "--port", str(LM_PORT)],
        cwd=LM_DIR,
        env={**os.environ,
             "LM_MODEL_ID": "ai-sage/Giga-Embeddings-instruct-480M-0826",
             "LM_REVISION": "1763d603adac8057bd6482a708001b0ed2e0a903",
             "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if httpx.get(f"{LM_URL}/v1/health", timeout=3).json()["status"] == "ok":
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            pytest.fail("local-models did not become ready")
        rc = redis_sync.Redis.from_url(REDIS_URL, decode_responses=True)
        for _ in range(60):
            try:
                if rc.ping():
                    break
            except redis_sync.ConnectionError:
                time.sleep(0.3)
        qc = create_sync_qdrant_client(QDRANT_URL)
        for _ in range(60):
            try:
                qc.get_collections()
                break
            except Exception:
                time.sleep(0.3)
        yield {"rc": rc, "qc": qc}
        rc.close()
    finally:
        lm.terminate()
        lm.wait(timeout=20)
        for name in ("b1-test-redis-s7", "b1-test-qdrant-s7"):
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _settings(docs_root: Path, **overrides: Any) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, documents_dir=str(docs_root), redis_url=REDIS_URL,
        qdrant_url=QDRANT_URL, local_models_url=LM_URL,
        chunk_tokens=30, chunk_overlap_tokens=5,  # many chunks -> multiple batches
        **overrides,
    )


BIG_PAGES = [
    f"{p}. Section\n" + " ".join(
        f"Clause {i}: the supplier must deliver goods of proper quality on time."
        for i in range(6))
    for p in range(1, 13)
]
SMALL_PAGES = [
    "1. Subject\nThe supplier delivers goods within ten working days of payment.",
]


async def _seed(docs_root: Path, pages: list[str]) -> tuple[str, str]:
    repo = RedisJobRepository(create_redis_client(REDIS_URL))
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
    (doc_dir / "original.pdf").write_bytes(make_text_pdf(pages))
    return job_id, doc_id


class _DyingRepo(QdrantIndexRepository):
    """Fails hard on the N-th upsert batch, simulating worker death mid-run."""

    def __init__(self, client: QdrantClient, collection: str, die_on: int) -> None:
        super().__init__(client, collection)
        self._die_on = die_on
        self._calls = 0

    def upsert_chunk_batch(self, **kwargs: Any) -> None:
        self._calls += 1
        if self._calls == self._die_on:
            raise _WorkerDied("simulated worker death")
        super().upsert_chunk_batch(**kwargs)


class _WorkerDied(BaseException):  # BaseException: bypasses IndexingError handling
    pass


class _CountingEmbedder(LocalModelsSyncClient):
    def __init__(self) -> None:
        super().__init__(LM_URL, timeout_s=120)
        self.embed_calls = 0

    def embed(self, request: Any) -> Any:
        self.embed_calls += 1
        return super().embed(request)


def _deps(qc: QdrantClient) -> dict[str, Any]:
    return {
        "index_repo": QdrantIndexRepository(qc, "document_chunks"),
        "embedder": _CountingEmbedder(),
    }


async def test_worker_killed_mid_run_then_recovered(
    services: dict[str, Any], tmp_path: Path
) -> None:
    rc, qc = services["rc"], services["qc"]
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)

    # a healthy neighbour doc that must survive every cleanup untouched
    nb_job, nb_doc = await _seed(tmp_path, SMALL_PAGES)
    assert run_index(lifecycle, nb_job, settings, **_deps(qc)) == "succeeded"
    nb_run = lifecycle.get_job(nb_job)["accepted_run_id"]  # type: ignore[index]
    nb_repo = QdrantIndexRepository(qc, "document_chunks")
    nb_count = nb_repo.count_run(nb_doc, nb_run)
    assert nb_count > 0

    # the victim: worker dies after the FIRST upsert batch (acceptance #1)
    job_id, doc_id = await _seed(tmp_path, BIG_PAGES)
    dying = _DyingRepo(qc, "document_chunks", die_on=2)
    lifecycle.transition(job_id, expected=["accepted"], new_status="queued")
    with pytest.raises(_WorkerDied):
        # drive the task body directly; death bypasses all compensation
        run_index(lifecycle, job_id, settings, index_repo=dying,
                  embedder=_CountingEmbedder())
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "running"  # crashed mid-flight
    partial_run = job["ingestion_run_id"]
    assert nb_repo.count_run(doc_id, partial_run) > 0  # partial points exist
    rc.delete(keys.lease_key(job_id))  # lease expires after the crash

    # reconciler requeues (the run is NOT complete -> no forward recovery)
    counters = reconcile(rc, lifecycle, nb_repo, store, settings)
    assert counters["requeued"] == 1
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "retry_wait"
    assert rc.lrange(keys.outbox_pending_key(), 0, -1).count(job_id) == 1

    # redelivery: the project recovers the job with a NEW run
    assert run_index(lifecycle, job_id, settings, **_deps(qc)) == "succeeded"
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "succeeded"
    accepted = job["accepted_run_id"]
    assert accepted != partial_run

    # reconciler removes the stale partial run — and ONLY it (acceptance #8)
    counters = reconcile(rc, lifecycle, nb_repo, store, settings)
    assert counters["stale_runs_removed"] >= 1
    assert nb_repo.count_run(doc_id, partial_run) == 0
    assert nb_repo.count_run(doc_id, accepted) == int(job["chunk_count"])
    assert nb_repo.count_run(nb_doc, nb_run) == nb_count  # neighbour untouched


async def test_crash_after_head_before_cas_forward_recovered(
    services: dict[str, Any], tmp_path: Path
) -> None:
    rc, qc = services["rc"], services["qc"]
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)
    job_id, doc_id = await _seed(tmp_path, SMALL_PAGES)
    lifecycle.transition(job_id, expected=["accepted"], new_status="queued")
    lifecycle.transition(job_id, expected=["queued"], new_status="running")
    lifecycle.incr_attempt(job_id)

    # run the FULL indexing (incl. head run_complete) but die before the CAS
    from app.ingestion.chunker import ApproxTokenLength
    from app.ingestion.pipeline import parse_document, save_parse_artifacts

    approx = ApproxTokenLength()
    doc_dir = tmp_path / doc_id
    result = parse_document(doc_dir=doc_dir, content_type="application/pdf",
                            max_pages=100, chunk_tokens=30, overlap_tokens=5,
                            length_fn=approx, length_fn_id=approx.id)
    save_parse_artifacts(doc_dir, result, chunk_tokens=30, overlap_tokens=5)
    lifecycle.set_fields(job_id, {"chunk_count": str(len(result.chunks))})
    token = lifecycle.acquire_lease(job_id, ttl_ms=60_000)
    assert token is not None
    outcome = run_indexing(
        job=lifecycle.get_job(job_id),  # type: ignore[arg-type]
        job_id=job_id, lifecycle=lifecycle, lease_token=token,
        index_repo=QdrantIndexRepository(qc, "document_chunks"),
        embedder=_CountingEmbedder(), bm25=_get_bm25(settings), settings=settings,
    )
    # CRASH HERE: no publish; the lease dies with the worker
    rc.delete(keys.lease_key(job_id))
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "running"

    counters = reconcile(rc, lifecycle, QdrantIndexRepository(qc, "document_chunks"),
                         store, settings)
    assert counters["forward_recovered"] == 1  # verified run published, not redone
    job = lifecycle.get_job(job_id)
    assert job is not None
    assert job["status"] == "succeeded"
    assert job["accepted_run_id"] == outcome.run_id


async def test_stale_token_and_terminal_cas_rejections(
    services: dict[str, Any], tmp_path: Path
) -> None:
    rc, qc = services["rc"], services["qc"]
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    job_id, _doc = await _seed(tmp_path, SMALL_PAGES)
    assert run_index(lifecycle, job_id, settings, **_deps(qc)) == "succeeded"
    # an old worker with a stale token can never publish its run (acceptance #3)
    stale = lifecycle.publish(job_id, token="stale-token", run_id="old-run")
    assert stale.verdict in ("STALE_TOKEN", "BAD_STATE")
    job = lifecycle.get_job(job_id)
    assert job is not None and job["accepted_run_id"] != "old-run"


async def test_compensation_unavailable_becomes_cleanup_pending(
    services: dict[str, Any], tmp_path: Path
) -> None:
    rc, qc = services["rc"], services["qc"]
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    store = FileStore(tmp_path, max_bytes=1 << 30)
    job_id, doc_id = await _seed(tmp_path, BIG_PAGES)
    lifecycle.transition(job_id, expected=["accepted"], new_status="queued")

    class _BrokenCleanupRepo(QdrantIndexRepository):
        def __init__(self) -> None:
            super().__init__(qc, "document_chunks")
            self._calls = 0

        def upsert_chunk_batch(self, **kwargs: Any) -> None:
            self._calls += 1
            if self._calls == 2:
                raise RuntimeError("qdrant went away mid-run")
            super().upsert_chunk_batch(**kwargs)

        def delete_run(self, doc_id: str, run_id: str) -> None:
            raise RuntimeError("qdrant still unavailable")  # compensation fails

    outcome = run_index(lifecycle, job_id, settings,
                        index_repo=_BrokenCleanupRepo(),
                        embedder=_CountingEmbedder())
    assert outcome.startswith("indexing_failed:")
    job = lifecycle.get_job(job_id)
    assert job is not None
    assert job["status"] == "cleanup_pending"  # NOT succeeded, NOT silently failed
    partial_run = job["ingestion_run_id"]
    repo = QdrantIndexRepository(qc, "document_chunks")
    assert repo.count_run(doc_id, partial_run) > 0  # garbage still there

    # later, compensation becomes available again -> reconciler finishes it
    counters = reconcile(rc, lifecycle, repo, store, settings)
    assert counters["cleaned"] == 1
    job = lifecycle.get_job(job_id)
    assert job is not None and job["status"] == "failed"
    assert repo.count_run(doc_id, partial_run) == 0


async def test_redelivery_of_completed_job_never_reembeds(
    services: dict[str, Any], tmp_path: Path
) -> None:
    rc, qc = services["rc"], services["qc"]
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path)
    job_id, _doc = await _seed(tmp_path, SMALL_PAGES)
    first = _CountingEmbedder()
    assert run_index(lifecycle, job_id, settings,
                     index_repo=QdrantIndexRepository(qc, "document_chunks"),
                     embedder=first) == "succeeded"
    assert first.embed_calls > 0

    second = _CountingEmbedder()
    outcome = run_index(lifecycle, job_id, settings,
                        index_repo=QdrantIndexRepository(qc, "document_chunks"),
                        embedder=second)
    assert outcome == "duplicate:succeeded"
    assert second.embed_calls == 0  # acceptance #5: embeddings never recomputed
