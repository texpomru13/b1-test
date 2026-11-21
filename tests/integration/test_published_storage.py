"""Public QA reads only the accepted Redis/Qdrant run. CI provides both services."""
from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
from app.config.settings import Settings
from app.domain.index_fingerprint import fingerprint_for_embedding
from app.inference.contract import EmbedResponse
from app.main import create_app
from app.qa.workflow import QaDependencies
from app.retrieval.bm25 import SparseVector
from app.storage.job_lifecycle_sync import SyncJobLifecycle, create_sync_client
from app.storage.qdrant_repo import (
    FingerprintMismatchError,
    QdrantIndexRepository,
    create_qdrant_client,
    create_sync_qdrant_client,
    point_id,
)
from app.storage.redis_repo import RedisJobRepository, create_redis_client

from tests.unit.test_qa_workflow import FakeBm25, FakeLM, ScriptedAdapter


async def test_only_published_run_can_answer_and_survives_cleanup() -> None:
    redis_url = os.environ.get("B1_TEST_REDIS_URL")
    qdrant_url = os.environ.get("B1_TEST_QDRANT_URL")
    if not (redis_url or qdrant_url or os.environ.get("B1_REQUIRE_STORES")):
        pytest.skip("set B1_TEST_REDIS_URL and B1_TEST_QDRANT_URL")
    assert redis_url and qdrant_url
    collection = f"ci_{uuid.uuid4().hex}"
    settings = Settings(_env_file=None, redis_url=redis_url, qdrant_url=qdrant_url,
                        qdrant_collection=collection)
    index = QdrantIndexRepository(create_sync_qdrant_client(qdrant_url), collection)
    reader = create_qdrant_client(qdrant_url)
    registry = RedisJobRepository(create_redis_client(redis_url))
    rc = create_sync_client(redis_url)
    lifecycle = SyncJobLifecycle(rc)
    doc, job = uuid.uuid4().hex, uuid.uuid4().hex
    fp = fingerprint_for_embedding(
        EmbedResponse(model_id="enc", revision="r", vectors=[[0.1] * 4], dim=4), settings,
        length_fn_id="custom", chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens)
    app = create_app(settings)
    app.state.redis_repo = registry
    app.state.qa_init_lock = asyncio.Lock()
    app.state.qa_semaphore = asyncio.Semaphore(1)
    app.state.qa_runtime = QaDependencies(
        settings=settings, qdrant=reader, local_models=FakeLM(), bm25=FakeBm25(), length_fn=len,
        adapter=ScriptedAdapter([{"answerable": False, "claims": [],
                                  "missing_information": ["quantity"]}] * 4))
    url = f"/api/v1/documents/{doc}/questions"
    question = {"question": "Сколько товара нужно поставить?", "debug": True}

    try:
        await registry.submit(idem_key=job, fingerprint=fp, doc_id=doc,
                              job_id=job, job_fields={"title": "test"})
        token = lifecycle.acquire_lease(job, ttl_ms=60_000)
        assert token and lifecycle.transition(
            job, expected=["accepted"], new_status="running", token=token,
            extra={"index_fingerprint": fp}).ok
        index.ensure_collection(dense_dim=4, computed_fingerprint=fp)
        for target, run, seq, fingerprint in [
            (doc, "accepted", 0, fp), (doc, "partial", 0, fp), (doc, "accepted", 1, "other"),
            ("neighbour", "accepted", 0, fp),
        ]:
            index.upsert_chunk_batch(
                doc_id=target, run_id=run,
                batch=[{"doc_id": target, "ingestion_run_id": run, "index_fingerprint": fingerprint,
                        "chunk_index": seq, "char_start": 0, "char_end": 5, "text": "Товар",
                        "section": None, "page_from": 1, "page_to": 1, "is_document_head": True}],
                dense_vectors=[[0.1] * 4], sparse_vectors=[SparseVector([1], [1.0])])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            # Even the verified Qdrant head remains invisible until Redis publication.
            index.mark_run_complete_on_head(doc, "accepted")
            assert (await client.post(url, json=question)).status_code == 409
            assert lifecycle.publish(job, token=token, run_id="accepted").ok
            response = await client.post(url, json=question)
            assert response.status_code == 200, response.text
            diag = response.json()["diagnostics"]
            assert diag["dense_returned"] == diag["sparse_returned"] == 1
            assert diag["context_chunk_ids"] == [point_id(doc, "accepted", 0)]
        index.delete_run(doc, "partial")
        assert index.count_run(doc, "partial") == 0
        assert index.count_run(doc, "accepted") == 2
        assert index.count_run("neighbour", "accepted") == 1
        with pytest.raises(FingerprintMismatchError):
            index.ensure_collection(dense_dim=4, computed_fingerprint="changed")
        assert index.head_is_complete(doc, "accepted")
    finally:
        await registry.aclose()
        await reader.close()
        if index._client.collection_exists(collection):
            index._client.delete_collection(collection)
        index.close()
        rc.close()
