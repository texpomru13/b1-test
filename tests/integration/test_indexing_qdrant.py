"""Stage-4 acceptance on REAL services: ephemeral Redis + Qdrant containers and
the host local-models service (Giga-480M at 127.0.0.1:18081).

Covers: points carry BOTH vectors and all mandatory payload fields; BM25 IDF +
query/document encoding; unpublished runs invisible to the QA scope; partial
batch failure never publishes; stale runs never mix with the accepted run;
schema/fingerprint mismatch refuses WITHOUT deleting data.
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
from app.jobs.tasks import run_index
from app.retrieval.bm25 import Bm25Adapter
from app.storage.job_lifecycle_sync import SyncJobLifecycle
from app.storage.qdrant_repo import (
    QdrantIndexRepository,
    create_sync_qdrant_client,
)
from app.storage.redis_repo import RedisJobRepository, create_redis_client
from qdrant_client import QdrantClient, models

from tests.helpers.pdfgen import make_text_pdf

REDIS_IMAGE = (
    "redis:8.10.3@sha256:1f88291b4c57bdf249f02eb085a8669889b2d2954e91d4de5118a5da6b78dd53"
)
QDRANT_IMAGE = (
    "qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
)
REDIS_PORT, QDRANT_PORT = 6414, 6415
REDIS_URL = f"redis://127.0.0.1:{REDIS_PORT}/0"
QDRANT_URL = f"http://127.0.0.1:{QDRANT_PORT}"
LM_URL = "http://127.0.0.1:18081"

MANDATORY_PAYLOAD_FIELDS = {
    "doc_id", "doc_title", "chunk_index", "page_from", "page_to", "section", "text",
    "char_start", "char_end", "ingestion_run_id", "index_fingerprint", "source_sha256",
    "created_at", "page_count", "chunk_count", "is_document_head", "run_complete",
    "segments",
}


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True, check=False
    ).returncode == 0


def _lm_ok() -> bool:
    try:
        return httpx.get(f"{LM_URL}/v1/health", timeout=3).json().get("status") == "ok"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_docker_ok() and _lm_ok()),
    reason="requires Docker and the host local-models service on :18081",
)


def _run_container(name: str, image: str, port: int, internal: int) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{port}:{internal}", image],
        check=True, capture_output=True,
    )


@pytest.fixture(scope="module")
def services() -> Iterator[tuple[redis_sync.Redis, QdrantClient]]:
    _run_container("b1-test-redis-s4", REDIS_IMAGE, REDIS_PORT, 6379)
    _run_container("b1-test-qdrant-s4", QDRANT_IMAGE, QDRANT_PORT, 6333)
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
    yield rc, qc
    rc.close()
    subprocess.run(["docker", "rm", "-f", "b1-test-redis-s4"], capture_output=True, check=False)
    subprocess.run(["docker", "rm", "-f", "b1-test-qdrant-s4"], capture_output=True, check=False)


def _settings(docs_root: Path, collection: str, **overrides: Any) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        documents_dir=str(docs_root),
        redis_url=REDIS_URL,
        qdrant_url=QDRANT_URL,
        local_models_url=LM_URL,
        qdrant_collection=collection,
        **overrides,
    )


async def _seed_job(docs_root: Path, pages: list[str]) -> tuple[str, str]:
    repo = RedisJobRepository(create_redis_client(REDIS_URL))
    job_id, doc_id = uuid.uuid4().hex, uuid.uuid4().hex
    result = await repo.submit(
        idem_key=f"k-{uuid.uuid4().hex[:8]}", fingerprint=uuid.uuid4().hex,
        job_id=job_id, doc_id=doc_id,
        job_fields={"title": "Договор", "file_sha256": "deadbeef",
                    "content_type": "application/pdf"},
    )
    await repo.aclose()
    assert result.verdict == "CREATED"
    doc_dir = docs_root / doc_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "original.pdf").write_bytes(make_text_pdf(pages))
    return job_id, doc_id


PAGES_SMALL = [
    "1. Subject\nThe supplier delivers goods within ten working days of payment.",
    "4.2 Penalties\nThe penalty is 0.1 percent per day, no more than ten percent total.",
]


async def test_full_indexing_publishes_with_both_vectors(
    services: tuple[redis_sync.Redis, QdrantClient], tmp_path: Path
) -> None:
    rc, qc = services
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path, "document_chunks")
    job_id, doc_id = await _seed_job(tmp_path, PAGES_SMALL)

    outcome = run_index(lifecycle, job_id, settings)
    assert outcome == "succeeded"

    record = lifecycle.get_job(job_id)
    assert record is not None
    assert record["status"] == "succeeded"
    run_id = record["accepted_run_id"]
    assert run_id
    chunk_count = int(record["chunk_count"])

    repo = QdrantIndexRepository(qc, "document_chunks")
    assert repo.count_run(doc_id, run_id) == chunk_count

    head = repo.retrieve_head(doc_id, run_id)
    assert head is not None
    payload = head.payload or {}
    assert MANDATORY_PAYLOAD_FIELDS.issubset(payload.keys())
    assert payload["is_document_head"] is True
    assert payload["run_complete"] is True  # flipped after count verification
    assert payload["chunk_index"] == 0 and payload["text"].strip()  # REAL text chunk
    vectors = head.vector or {}
    assert len(vectors["dense"]) == 1024
    assert len(vectors["bm25"].indices) > 0

    info = qc.get_collection("document_chunks")
    assert info.config.params.sparse_vectors["bm25"].modifier == models.Modifier.IDF

    # BM25 doc/query encoding: symmetric synthetic token for an exact designation
    bm25 = Bm25Adapter(language="russian", avg_len=256.0)
    doc_vec = bm25.embed_documents(["Пункт 4.2.1: неустойка 0,1% в день"])[0]
    query_vec = bm25.embed_query("что говорит пункт 4.2.1?")
    assert set(doc_vec.indices) & set(query_vec.indices)
    assert all(value == 1.0 for value in query_vec.values)  # query side: unit weights


class _FailingRepo(QdrantIndexRepository):
    def __init__(self, client: QdrantClient, collection: str, fail_on_batch: int) -> None:
        super().__init__(client, collection)
        self._fail_on = fail_on_batch
        self._calls = 0

    def upsert_chunk_batch(self, **kwargs: Any) -> None:
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("injected batch failure")
        super().upsert_chunk_batch(**kwargs)


async def test_partial_batch_failure_never_publishes(
    services: tuple[redis_sync.Redis, QdrantClient], tmp_path: Path
) -> None:
    rc, qc = services
    lifecycle = SyncJobLifecycle(rc)
    # ~70+ chunks so at least two upsert batches of 64 are needed
    paragraphs = " ".join(
        f"Clause {i}: the supplier must deliver goods of proper quality on time." for i in range(6)
    )
    pages = [f"{p}. Section\n{paragraphs}" for p in range(1, 13)]
    collection = "partial_fail_test"  # own collection: different chunk params
    settings = _settings(tmp_path, collection, chunk_tokens=30,
                         chunk_overlap_tokens=5)
    job_id, doc_id = await _seed_job(tmp_path, pages)

    failing = _FailingRepo(qc, collection, fail_on_batch=2)
    outcome = run_index(lifecycle, job_id, settings, index_repo=failing)
    assert outcome.startswith("indexing_failed:")

    record = lifecycle.get_job(job_id)
    assert record is not None
    assert record["status"] == "failed"
    assert not record.get("accepted_run_id")  # never published
    failed_run = record["ingestion_run_id"]
    repo = QdrantIndexRepository(qc, collection)
    assert repo.count_run(doc_id, failed_run) == 0  # incomplete run cleaned up
    # QA scope rule: no accepted pointer -> nothing is served for this doc
    complete_heads, _ = qc.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="run_complete", match=models.MatchValue(value=True)),
        ]),
        limit=10,
    )
    assert complete_heads == []


async def test_stale_run_never_mixes_with_accepted(
    services: tuple[redis_sync.Redis, QdrantClient], tmp_path: Path
) -> None:
    rc, qc = services
    lifecycle = SyncJobLifecycle(rc)
    settings = _settings(tmp_path, "document_chunks")
    job_id, doc_id = await _seed_job(tmp_path, PAGES_SMALL)
    assert run_index(lifecycle, job_id, settings) == "succeeded"
    record = lifecycle.get_job(job_id)
    assert record is not None
    accepted_run = record["accepted_run_id"]
    repo = QdrantIndexRepository(qc, "document_chunks")
    accepted_count = repo.count_run(doc_id, accepted_run)

    # stale worker writes a point of an OLD run after losing the lease
    stale_run = uuid.uuid4().hex
    qc.upsert(
        collection_name="document_chunks",
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": [0.01] * 1024,
                    "bm25": models.SparseVector(indices=[1], values=[1.0])},
            payload={"doc_id": doc_id, "ingestion_run_id": stale_run,
                     "chunk_index": 0, "is_document_head": True, "run_complete": True,
                     "text": "stale", "index_fingerprint": "stale"},
        )],
        wait=True,
    )
    # the accepted pointer in Redis is authoritative: scope = doc_id + accepted_run_id
    assert lifecycle.get_job(job_id)["accepted_run_id"] == accepted_run  # type: ignore[index]
    assert repo.count_run(doc_id, accepted_run) == accepted_count  # unchanged
    scoped, _ = qc.scroll(
        collection_name="document_chunks",
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(
                key="ingestion_run_id", match=models.MatchValue(value=accepted_run)
            ),
        ]),
        limit=100, with_payload=["ingestion_run_id"],
    )
    assert all(p.payload["ingestion_run_id"] == accepted_run for p in scoped)


async def test_schema_mismatch_refused_without_deletion(
    services: tuple[redis_sync.Redis, QdrantClient], tmp_path: Path
) -> None:
    rc, qc = services
    lifecycle = SyncJobLifecycle(rc)
    collection = "mismatch_dim_test"
    qc.create_collection(
        collection_name=collection,
        vectors_config={"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    qc.upsert(collection_name=collection, points=[models.PointStruct(
        id=str(uuid.uuid4()), vector={"dense": [0.1] * 8}, payload={"keep": True})],
        wait=True)
    settings = _settings(tmp_path, collection)
    job_id, _doc_id = await _seed_job(tmp_path, PAGES_SMALL)

    outcome = run_index(lifecycle, job_id, settings)
    assert outcome == "indexing_failed:INDEX_SCHEMA_MISMATCH"
    record = lifecycle.get_job(job_id)
    assert record is not None and record["error_code"] == "INDEX_SCHEMA_MISMATCH"
    assert qc.count(collection_name=collection, exact=True).count == 1  # data intact


async def test_fingerprint_mismatch_refused(
    services: tuple[redis_sync.Redis, QdrantClient], tmp_path: Path
) -> None:
    rc, qc = services
    lifecycle = SyncJobLifecycle(rc)
    collection = "mismatch_fp_test"
    qc.create_collection(
        collection_name=collection,
        vectors_config={"dense": models.VectorParams(
            size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    qc.upsert(collection_name=collection, points=[models.PointStruct(
        id=str(uuid.uuid4()),
        vector={"dense": [0.01] * 1024,
                "bm25": models.SparseVector(indices=[1], values=[1.0])},
        payload={"index_fingerprint": "an-older-embedding-space"})], wait=True)
    settings = _settings(tmp_path, collection)
    job_id, _doc_id = await _seed_job(tmp_path, PAGES_SMALL)

    outcome = run_index(lifecycle, job_id, settings)
    assert outcome == "indexing_failed:INDEX_FINGERPRINT_MISMATCH"
    assert qc.count(collection_name=collection, exact=True).count == 1  # intact
