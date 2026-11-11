"""Stage-5 acceptance on REAL services: ephemeral Qdrant + two host local-models
instances (Giga-480M encoder with a Qwen3-Reranker-0.6B yes/no reranker, and a
BGE cross-encoder variant). Russian twin contracts share clause numbers but
differ in percentages — exact/paraphrase retrieval and scope isolation are
verified on real embeddings, real BM25 and a real mandatory reranker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from app.config.settings import Settings
from app.inference.client import LocalModelsClient
from app.ingestion.canonical import build_canonical
from app.ingestion.chunker import ApproxTokenLength, split_into_chunks
from app.ingestion.pdf_loader import ExtractedPage
from app.ingestion.sections import detect_sections
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import RetrievalScope, hybrid_search
from app.storage.qdrant_repo import (
    QdrantIndexRepository,
    create_sync_qdrant_client,
    point_id,
)
from qdrant_client import AsyncQdrantClient

QDRANT_IMAGE = (
    "qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2"
)
QDRANT_PORT = 6416
QDRANT_URL = f"http://127.0.0.1:{QDRANT_PORT}"
LM_QWEN_PORT, LM_BGE_PORT = 18082, 18083
COLLECTION = "retrieval_stage5"
FP = "fp-stage5-test"

LM_DIR = Path(__file__).resolve().parents[2] / "services" / "local-models"

ENV_COMMON = {
    "LM_MODEL_ID": "ai-sage/Giga-Embeddings-instruct-480M-0826",
    "LM_REVISION": "1763d603adac8057bd6482a708001b0ed2e0a903",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "info"], capture_output=True, check=False
    ).returncode == 0


def _cached(repo: str) -> bool:
    return (Path.home() / ".cache/huggingface/hub" / repo).exists()


pytestmark = pytest.mark.skipif(
    not (
        _docker_ok()
        and (LM_DIR / ".venv").exists()
        and _cached("models--Qwen--Qwen3-Reranker-0.6B")
        and _cached("models--BAAI--bge-reranker-v2-m3")
    ),
    reason="requires Docker, the local-models venv and cached reranker models",
)


def _start_lm(port: int, extra_env: dict[str, str]) -> subprocess.Popen[bytes]:
    env = {**os.environ, **ENV_COMMON, **extra_env}
    return subprocess.Popen(
        [str(LM_DIR / ".venv/bin/uvicorn"), "app_lm.server:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=LM_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wait_lm(port: int, timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            body = httpx.get(f"http://127.0.0.1:{port}/v1/health", timeout=3).json()
            if body["status"] == "ok":
                return
            if body["status"] == "error":
                pytest.fail(f"local-models on :{port} errored: {body}")
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.fail(f"local-models on :{port} did not become ready")


@pytest.fixture(scope="module")
def stack() -> Iterator[dict[str, Any]]:
    subprocess.run(["docker", "rm", "-f", "b1-test-qdrant-s5"], capture_output=True,
                   check=False)
    subprocess.run(
        ["docker", "run", "-d", "--name", "b1-test-qdrant-s5",
         "-p", f"{QDRANT_PORT}:6333", QDRANT_IMAGE],
        check=True, capture_output=True,
    )
    qwen = _start_lm(LM_QWEN_PORT, {
        "LM_RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
        "LM_RERANKER_REVISION": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "LM_RERANKER_ADAPTER": "qwen_yes_no",
    })
    bge = _start_lm(LM_BGE_PORT, {
        "LM_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "LM_RERANKER_REVISION": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "LM_RERANKER_ADAPTER": "cross_encoder",
    })
    try:
        _wait_lm(LM_QWEN_PORT)
        _wait_lm(LM_BGE_PORT)
        sync_qc = create_sync_qdrant_client(QDRANT_URL)
        for _ in range(60):
            try:
                sync_qc.get_collections()
                break
            except Exception:
                time.sleep(0.3)
        yield {"sync_qc": sync_qc}
    finally:
        qwen.terminate()
        bge.terminate()
        qwen.wait(timeout=20)
        bge.wait(timeout=20)
        subprocess.run(["docker", "rm", "-f", "b1-test-qdrant-s5"],
                       capture_output=True, check=False)


CONTRACT_A_PAGES = [
    "Договор поставки № АБ-123/24\n\n"
    "1. Предмет договора\n"
    "Поставщик обязуется поставить Покупателю товар надлежащего качества, "
    "а Покупатель обязуется принять и оплатить его в установленный срок.\n\n"
    "3. Срок поставки\n"
    "Поставка осуществляется в течение десяти рабочих дней с момента оплаты счёта.",
    "5. Ответственность сторон\n\n"
    "5.1 За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку "
    "в размере 0,1% от стоимости товара за каждый день просрочки, "
    "но не более 10% от общей цены договора.\n\n"
    "5.2 Уплата неустойки не освобождает стороны от исполнения обязательств.",
]

CONTRACT_B_PAGES = [
    "Договор поставки № ВГ-456/24\n\n"
    "1. Предмет договора\n"
    "Поставщик обязуется поставить Покупателю оборудование, а Покупатель — "
    "оплатить его на условиях предоплаты.\n\n"
    "3. Срок поставки\n"
    "Поставка осуществляется в течение тридцати календарных дней.",
    "5. Ответственность сторон\n\n"
    "5.1 За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку "
    "в размере 0,5% от стоимости оборудования за каждый день просрочки, "
    "но не более 20% от общей цены договора.\n\n"
    "5.2 Споры разрешаются в арбитражном суде.",
]


class CountingQdrant(AsyncQdrantClient):
    def __init__(self, url: str) -> None:
        super().__init__(url=url, timeout=20)
        self.query_calls = 0
        self.batch_calls = 0

    async def query_points(self, *args: Any, **kwargs: Any) -> Any:
        self.query_calls += 1
        return await super().query_points(*args, **kwargs)

    async def query_batch_points(self, *args: Any, **kwargs: Any) -> Any:
        self.batch_calls += 1
        return await super().query_batch_points(*args, **kwargs)


async def _index_contract(
    stack: dict[str, Any], doc_id: str, run_id: str, pages: list[str]
) -> dict[str, Any]:
    """Index via the app's own canonical/section/chunk + embed + BM25 code paths."""
    extracted = [
        ExtractedPage(page=i + 1, text=text, has_text=True)
        for i, text in enumerate(pages)
    ]
    canonical = build_canonical(extracted)
    sections = detect_sections(canonical.text)
    chunks = split_into_chunks(
        canonical, sections, chunk_tokens=120, overlap_tokens=20,
        length_fn=ApproxTokenLength(),
    )
    assert len(chunks) >= 3
    lm = LocalModelsClient(f"http://127.0.0.1:{LM_QWEN_PORT}", timeout_s=120)
    from app.inference.contract import EmbedRequest, InputType

    dense = (
        await lm.embed(EmbedRequest(texts=[c.text for c in chunks],
                                    input_type=InputType.DOCUMENT))
    ).vectors
    await lm.aclose()
    bm25 = Bm25Adapter(language="russian", avg_len=256.0)
    sparse = bm25.embed_documents([c.text for c in chunks])

    repo = QdrantIndexRepository(stack["sync_qc"], COLLECTION)
    repo.ensure_collection(dense_dim=len(dense[0]), computed_fingerprint=FP)
    payloads = [
        {
            "doc_id": doc_id, "doc_title": "Договор", "chunk_index": c.index,
            "page_from": c.page_from, "page_to": c.page_to, "section": c.section,
            "text": c.text, "char_start": c.char_start, "char_end": c.char_end,
            "ingestion_run_id": run_id, "index_fingerprint": FP,
            "source_sha256": "test", "created_at": "t", "page_count": len(pages),
            "chunk_count": len(chunks), "is_document_head": c.index == 0,
            "run_complete": False, "segments": [],
        }
        for c in chunks
    ]
    repo.upsert_chunk_batch(doc_id=doc_id, run_id=run_id, batch=payloads,
                            dense_vectors=dense, sparse_vectors=sparse)
    repo.mark_run_complete_on_head(doc_id, run_id)
    gold = next(c for c in chunks if "0,1%" in c.text or "0,5%" in c.text)
    return {
        "chunks": chunks,
        "point_ids": {point_id(doc_id, run_id, c.index) for c in chunks},
        "gold_chunk_index": gold.index,
        "n": len(chunks),
    }


def _settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None, qdrant_url=QDRANT_URL, qdrant_collection=COLLECTION,
        local_models_url=f"http://127.0.0.1:{LM_QWEN_PORT}",
    )


@pytest.fixture(scope="module")
def corpus(stack: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    doc_a, doc_b = uuid.uuid4().hex, uuid.uuid4().hex
    run_a, run_b = uuid.uuid4().hex, uuid.uuid4().hex
    info_a = asyncio.run(_index_contract(stack, doc_a, run_a, CONTRACT_A_PAGES))
    info_b = asyncio.run(_index_contract(stack, doc_b, run_b, CONTRACT_B_PAGES))
    return {
        "a": {"scope": RetrievalScope(doc_a, run_a, FP), **info_a},
        "b": {"scope": RetrievalScope(doc_b, run_b, FP), **info_b},
    }


async def _search(corpus_entry: dict[str, Any], question: str, *, debug: bool = False):
    qdrant = CountingQdrant(QDRANT_URL)
    lm = LocalModelsClient(f"http://127.0.0.1:{LM_QWEN_PORT}", timeout_s=300)
    bm25 = Bm25Adapter(language="russian", avg_len=256.0)
    try:
        result = await hybrid_search(
            question=question, scope=corpus_entry["scope"], settings=_settings(),
            qdrant=qdrant, local_models=lm, bm25=bm25,
            length_fn=ApproxTokenLength(), debug=debug,
        )
    finally:
        await lm.aclose()
        await qdrant.close()
    return result, qdrant


async def test_exact_clause_query_finds_gold_span(corpus: dict[str, Any]) -> None:
    entry = corpus["a"]
    result, qdrant = await _search(entry, "Какая неустойка предусмотрена пунктом 5.1?")
    assert qdrant.query_calls == 1  # server-side fusion: ONE query, no Python merge
    assert qdrant.batch_calls == 0
    assert result.context, "expected non-empty context"
    texts = [c.text for c in result.context]
    assert any("0,1%" in t for t in texts), texts
    # gold chunk is the top-ranked one
    assert result.context[0].chunk_index == entry["gold_chunk_index"]
    # every context chunk belongs to doc A's point set (scope isolation)
    assert all(c.chunk_id in entry["point_ids"] for c in result.context)


async def test_paraphrase_finds_gold_span(corpus: dict[str, Any]) -> None:
    entry = corpus["a"]
    result, _ = await _search(entry, "какой штраф грозит за просрочку поставки товара?")
    assert any(c.chunk_index == entry["gold_chunk_index"] for c in result.context)


async def test_similar_documents_are_isolated(corpus: dict[str, Any]) -> None:
    result_b, _ = await _search(corpus["b"], "Какая неустойка предусмотрена пунктом 5.1?")
    texts_b = " ".join(c.text for c in result_b.context)
    assert "0,5%" in texts_b
    assert "0,1%" not in texts_b  # doc A's value never leaks into doc B's scope
    assert all(c.chunk_id in corpus["b"]["point_ids"] for c in result_b.context)


async def test_debug_counts_real_on_real_qdrant(corpus: dict[str, Any]) -> None:
    entry = corpus["a"]
    result, qdrant = await _search(entry, "Каков срок поставки?", debug=True)
    diag = result.diagnostics
    assert qdrant.query_calls == 1 and qdrant.batch_calls == 1
    assert diag.branch_counts_source == "diagnostic_replay"
    # real branch counts equal the doc's chunk count (< the 60 limit — not echoed)
    assert diag.dense_returned == entry["n"] < 60
    assert 0 < diag.sparse_returned <= entry["n"]
    assert diag.fused_returned <= 60
    assert diag.reranked_count == diag.fused_returned
    assert diag.model_ids["reranker"] == "Qwen/Qwen3-Reranker-0.6B"
    assert set(diag.context_chunk_ids).issubset(entry["point_ids"])


def test_bge_cross_encoder_adapter_orders_pairs(stack: dict[str, Any]) -> None:
    response = httpx.post(
        f"http://127.0.0.1:{LM_BGE_PORT}/v1/rerank",
        json={"query": "Какая неустойка за просрочку поставки?",
              "documents": [
                  {"id": "rel", "text": "Неустойка составляет 0,1% за каждый день просрочки."},
                  {"id": "irr", "text": "Поставка осуществляется автомобильным транспортом."},
              ]},
        timeout=120,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "BAAI/bge-reranker-v2-m3"
    scores = {s["id"]: s["score"] for s in body["scores"]}
    assert scores["rel"] > scores["irr"]
