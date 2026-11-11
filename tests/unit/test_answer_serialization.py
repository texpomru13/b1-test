"""200 bodies of the questions route must validate against AnswerResponse itself
(stage-9 finding: exclude_none dropped the required-but-nullable `section`)."""

import httpx
from app.api.questions import router
from app.schemas.qa import AnswerResponse, Citation, QaDiagnostics
from fastapi import FastAPI


def _route_kwargs() -> dict:
    route = next(r for r in router.routes if getattr(r, "path", "").endswith("/questions"))
    return {
        "response_model": route.response_model,  # type: ignore[attr-defined]
        "response_model_exclude_none": route.response_model_exclude_none,  # type: ignore[attr-defined]
        "response_model_exclude_unset": route.response_model_exclude_unset,  # type: ignore[attr-defined]
    }


def _answer() -> AnswerResponse:
    return AnswerResponse(answer="Срок — 3 года. [1]", found=True, citations=[Citation(
        id=1, text="3 года", page=1, page_to=1, section=None, chunk_id="c1",
        char_start=0, char_end=6)])


def _diag() -> QaDiagnostics:
    return QaDiagnostics(
        dense_returned=None, sparse_returned=None, fused_returned=1, reranked_count=1,
        context_count=1, context_chunk_ids=["c1"], cited_chunk_ids=["c1"],
        rerank_scores={"c1": 0.9}, stage_ms={}, model_ids={}, token_usage={},
        branch_counts_source=None, diagnostic_warnings=[])


async def _get(debug: bool) -> dict:
    kwargs = _route_kwargs()
    app = FastAPI()

    @app.get("/q", **kwargs)
    async def q() -> AnswerResponse:
        resp = _answer()
        if debug:  # same mechanism as the real route
            resp = resp.model_copy(update={"diagnostics": _diag()})
        return resp

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        return (await client.get("/q")).json()


async def test_null_section_is_kept_and_body_validates() -> None:
    body = await _get(debug=False)
    assert "section" in body["citations"][0] and body["citations"][0]["section"] is None
    assert "diagnostics" not in body  # only with debug=true
    AnswerResponse.model_validate(body)


async def test_debug_diagnostics_keep_nullable_counts() -> None:
    body = await _get(debug=True)
    assert body["diagnostics"]["dense_returned"] is None
    assert "branch_counts_source" in body["diagnostics"]
    AnswerResponse.model_validate(body)
