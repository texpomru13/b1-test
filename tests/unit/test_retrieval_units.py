from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from app.config.settings import Settings
from app.inference.client import LocalModelsClient, LocalModelsError
from app.inference.contract import (
    EmbedResponse,
    RerankResponse,
    RerankScore,
)
from app.retrieval.bm25 import SparseVector
from app.retrieval.search import (
    RetrievalError,
    RetrievalScope,
    effective_context_budget,
    hybrid_search,
    scope_filter,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


SCOPE = RetrievalScope(doc_id="doc-A", accepted_run_id="run-1", index_fingerprint="fp-1")


def test_scope_filter_has_all_three_conditions() -> None:
    flt = scope_filter(SCOPE)
    keys = [c.key for c in flt.must]
    assert keys == ["doc_id", "ingestion_run_id", "index_fingerprint"]


def test_effective_budget_math_and_floor() -> None:
    settings = make_settings()
    # 16384 - 1500 - q - 4096 - 2048 - 512 = 8228 - q; capped by CONTEXT_BUDGET=6000
    assert effective_context_budget(settings, question_tokens=10) == 6000
    tight = make_settings(local_context_tokens=9000, context_budget_tokens=6000,
                          llm_reasoning_budget_tokens=1500, llm_max_output_tokens=500)
    # 9000 - 1500(sys) - 100(q) - 1500 - 500 - 512 = 4888 < budget 6000
    assert effective_context_budget(tight, question_tokens=100) == 4888
    tiny = make_settings(local_context_tokens=8193, context_budget_tokens=6000,
                         llm_reasoning_budget_tokens=0, llm_max_output_tokens=100)
    assert effective_context_budget(tiny, question_tokens=10_000) == 0


def _point(pid: str, index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=pid, score=0.5,
        payload={"text": text, "chunk_index": index, "section": None,
                 "page_from": 1, "page_to": 1, "char_start": 0, "char_end": len(text)},
    )


class FakeQdrant:
    def __init__(self, points: list[SimpleNamespace]) -> None:
        self.points = points
        self.query_calls = 0
        self.batch_calls = 0
        self.fail_batch = False

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.query_calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(points=self.points)

    async def query_batch_points(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.batch_calls += 1
        if self.fail_batch:
            raise RuntimeError("replay down")
        return [SimpleNamespace(points=self.points[:2]), SimpleNamespace(points=self.points[:1])]


class FakeLM:
    def __init__(self, scores: dict[str, float], *, rerank_fails: bool = False) -> None:
        self._scores = scores
        self._rerank_fails = rerank_fails

    async def embed(self, request: Any) -> EmbedResponse:
        return EmbedResponse(vectors=[[0.1, 0.2]], model_id="enc", revision="r", dim=2)

    async def rerank(self, request: Any) -> RerankResponse:
        if self._rerank_fails:
            raise LocalModelsError("rerank endpoint down")
        return RerankResponse(
            scores=[RerankScore(id=d.id, score=self._scores[d.id]) for d in request.documents],
            model_id="rr", revision="r",
        )


class FakeBm25:
    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 2], values=[1.0, 1.0])


def char_len(text: str) -> int:
    return len(text)


async def test_single_fused_query_and_tie_break() -> None:
    points = [_point("b", 1, "x" * 10), _point("a", 2, "y" * 10), _point("c", 3, "z" * 10)]
    qdrant = FakeQdrant(points)
    lm = FakeLM({"a": 0.9, "b": 0.9, "c": 0.1})  # tie between a and b
    result = await hybrid_search(
        question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
        local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
    )
    assert qdrant.query_calls == 1  # exactly ONE fused query, no Python merge
    assert qdrant.batch_calls == 0  # no probes without debug
    # ties resolved by ascending id: a before b; c last
    assert [c.chunk_id for c in result.context] == ["a", "b", "c"]
    # the root query carries the scope filter and the fusion query
    assert qdrant.last_kwargs["query_filter"] is not None
    assert len(qdrant.last_kwargs["prefetch"]) == 2


async def test_budget_drops_whole_chunks_and_reflects_in_allowed() -> None:
    big = _point("big", 1, "x" * 5000)
    small1 = _point("s1", 2, "y" * 100)
    small2 = _point("s2", 3, "z" * 100)
    qdrant = FakeQdrant([big, small1, small2])
    lm = FakeLM({"big": 0.9, "s1": 0.8, "s2": 0.7})
    settings = make_settings(
        local_context_tokens=9000, context_budget_tokens=300,
        llm_reasoning_budget_tokens=0, llm_max_output_tokens=100,
    )
    result = await hybrid_search(
        question="q", scope=SCOPE, settings=settings, qdrant=qdrant,  # type: ignore[arg-type]
        local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
    )
    # 'big' (5000 chars) exceeds the 300-token budget and is dropped WHOLE
    assert [c.chunk_id for c in result.context] == ["s1", "s2"]
    assert result.allowed_chunk_ids == ["s1", "s2"]
    assert result.diagnostics.dropped_by_budget_chunk_ids == ["big"]
    assert result.diagnostics.context_count == 2


async def test_reranker_failure_is_error_not_dense_only() -> None:
    qdrant = FakeQdrant([_point("a", 1, "text")])
    lm = FakeLM({}, rerank_fails=True)
    with pytest.raises(RetrievalError) as excinfo:
        await hybrid_search(
            question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
            local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        )
    assert excinfo.value.code == "RERANKER_UNAVAILABLE"


async def test_debug_counts_are_real_not_limits() -> None:
    points = [_point(str(i), i, "t" * 10) for i in range(3)]
    qdrant = FakeQdrant(points)
    lm = FakeLM({str(i): 1.0 - i / 10 for i in range(3)})
    result = await hybrid_search(
        question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
        local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        debug=True,
    )
    diag = result.diagnostics
    assert qdrant.batch_calls == 1
    assert diag.branch_counts_source == "diagnostic_replay"
    assert diag.dense_returned == 2 and diag.sparse_returned == 1  # real lens, not 60
    assert diag.fused_returned == 3
    assert diag.cited_chunk_ids is None  # QA fills this; never mixed with context ids


async def test_debug_replay_failure_yields_null_and_warning() -> None:
    qdrant = FakeQdrant([_point("a", 1, "t")])
    qdrant.fail_batch = True
    lm = FakeLM({"a": 0.5})
    result = await hybrid_search(
        question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
        local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        debug=True,
    )
    diag = result.diagnostics
    assert diag.dense_returned is None and diag.sparse_returned is None
    assert diag.diagnostic_warnings  # explicit warning instead of fabricated numbers
    assert result.context  # retrieval itself unaffected


async def test_frozen_candidates_written(tmp_path: Path) -> None:
    qdrant = FakeQdrant([_point("a", 1, "t"), _point("b", 2, "u")])
    lm = FakeLM({"a": 0.9, "b": 0.1})
    await hybrid_search(
        question="вопрос", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
        local_models=lm, bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        freeze_dir=tmp_path,
    )
    files = list(tmp_path.glob("candidates-*.json"))  # noqa: ASYNC240 - test code
    assert len(files) == 1
    import json

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["doc_id"] == "doc-A"
    assert [c["fused_rank"] for c in payload["candidates"]] == [0, 1]


async def test_client_rerank_501_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(501, json={"error": "no reranker configured"})

    client = LocalModelsClient("http://lm")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://lm"
    )
    from app.inference.contract import RerankDocument, RerankRequest

    with pytest.raises(LocalModelsError):
        await client.rerank(
            RerankRequest(query="q", documents=[RerankDocument(id="1", text="t")])
        )


class FlakyQdrant(FakeQdrant):
    """Fails the first ``failures`` fused queries with the given exception."""

    def __init__(self, points: list[SimpleNamespace], *, failures: int,
                 error: Exception) -> None:
        super().__init__(points)
        self._failures = failures
        self._error = error

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        if self.query_calls < self._failures:
            self.query_calls += 1
            raise self._error
        return await super().query_points(**kwargs)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.reliability.policy.backoff_delay", lambda *a, **k: 0.0)


def _qdrant_connect_error() -> Exception:
    from qdrant_client.http.exceptions import ResponseHandlingException

    return ResponseHandlingException(httpx.ConnectError("connection refused"))


async def test_qdrant_transient_failure_is_retried(no_backoff: None) -> None:
    qdrant = FlakyQdrant([_point("a", 1, "text")], failures=1,
                         error=_qdrant_connect_error())
    result = await hybrid_search(
        question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
        local_models=FakeLM({"a": 0.9}), bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
    )
    assert qdrant.query_calls == 2  # one failed attempt + the successful one
    assert [c.chunk_id for c in result.context] == ["a"]


async def test_qdrant_outage_is_bounded_and_reported(no_backoff: None) -> None:
    qdrant = FlakyQdrant([_point("a", 1, "text")], failures=99,
                         error=_qdrant_connect_error())
    with pytest.raises(RetrievalError) as excinfo:
        await hybrid_search(
            question="q", scope=SCOPE, settings=make_settings(qdrant_max_attempts=3),
            qdrant=qdrant,  # type: ignore[arg-type]
            local_models=FakeLM({"a": 0.9}), bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        )
    assert excinfo.value.code == "QDRANT_UNAVAILABLE"  # an error, never found=false
    assert qdrant.query_calls == 3  # exactly QDRANT_MAX_ATTEMPTS, then give up


async def test_qdrant_bad_request_is_not_retried(no_backoff: None) -> None:
    from qdrant_client.http.exceptions import UnexpectedResponse

    bad_request = UnexpectedResponse(
        status_code=400, reason_phrase="Bad Request", content=b"{}",
        headers=httpx.Headers({}),
    )
    qdrant = FlakyQdrant([_point("a", 1, "text")], failures=99, error=bad_request)
    with pytest.raises(RetrievalError):
        await hybrid_search(
            question="q", scope=SCOPE, settings=make_settings(), qdrant=qdrant,  # type: ignore[arg-type]
            local_models=FakeLM({"a": 0.9}), bm25=FakeBm25(), length_fn=char_len,  # type: ignore[arg-type]
        )
    assert qdrant.query_calls == 1
