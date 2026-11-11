"""POST /api/v1/documents/{doc_id}/questions — independent questions, no memory.

Outcome separation (§7.3/§9.3): insufficient evidence -> 200 found=false;
model/reranker/runtime outage -> 503; deadline -> 504; invalid model output
after the single repair -> 502 MODEL_OUTPUT_INVALID. Infra errors are NEVER
masked as found=false.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Request

from app.domain.job import JobStatus
from app.ingestion.chunker import ApproxTokenLength, HFTokenizerLength
from app.providers.factory import (
    ProviderConfigError,
    ProviderUnavailable,
    build_chat_adapter,
)
from app.qa.assemble import assemble_answer
from app.qa.graph import (
    OUTCOME_ANSWERED,
    OUTCOME_NOT_FOUND,
    QaDependencies,
    build_qa_graph,
    run_qa,
)
from app.reliability.deadline import Deadline, DeadlineExceeded
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import RetrievalError, RetrievalScope
from app.schemas.errors import ApiError, ErrorCode
from app.schemas.qa import (
    NO_ANSWER_MESSAGE,
    AnswerResponse,
    QaDiagnostics,
    QuestionRequest,
)

router = APIRouter(prefix="/api/v1", tags=["questions"])
logger = structlog.get_logger("api.questions")


async def _qa_runtime(request: Request) -> dict[str, Any]:
    """Lazily build and cache the QA runtime (adapter, bm25, compiled graph)."""
    app = request.app
    if getattr(app.state, "qa_runtime", None) is None:
        settings = app.state.settings
        try:
            adapter = build_chat_adapter(settings)
        except ProviderConfigError as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE, str(exc),
                           retryable=False) from exc
        bm25 = await asyncio.to_thread(
            lambda: Bm25Adapter(
                language=settings.bm25_language,
                avg_len=float(settings.bm25_avg_len),
                cache_dir=settings.fastembed_cache_dir,
            )
        )
        length_fn: Any
        if settings.embedding_tokenizer_path:
            length_fn = HFTokenizerLength(
                __import__("pathlib").Path(settings.embedding_tokenizer_path)
            )
        else:
            length_fn = ApproxTokenLength()
        deps = QaDependencies(
            settings=settings,
            qdrant=app.state.qdrant_repo._client,
            local_models=app.state.local_models,
            bm25=bm25,
            length_fn=length_fn,
            adapter=adapter,
        )
        app.state.qa_runtime = {
            "graph": build_qa_graph(deps),
            "adapter": adapter,
        }
    return app.state.qa_runtime


# exclude_UNSET (not exclude_none): `diagnostics` is omitted unless debug=true, but
# nullable required fields (citation `section`, diagnostics branch counts) are kept
# as explicit nulls so every 200 body validates against the published schema.
@router.post("/documents/{doc_id}/questions", response_model=AnswerResponse,
             response_model_exclude_unset=True)
async def ask_question(
    request: Request, doc_id: str, body: QuestionRequest
) -> AnswerResponse:
    settings = request.app.state.settings
    repo = request.app.state.redis_repo

    if len(body.question) > settings.max_question_chars:
        raise ApiError(
            422, ErrorCode.VALIDATION_ERROR,
            f"body.question: must be at most {settings.max_question_chars} characters",
        )

    job_id = await repo.get_job_id_for_doc(doc_id)
    if not job_id:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"document '{doc_id}' is unknown")
    job = await repo.get_job(job_id)
    if not job:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"document '{doc_id}' is unknown")
    status = JobStatus(job["status"])
    if status != JobStatus.SUCCEEDED or not job.get("accepted_run_id"):
        detail = (
            "indexing failed; the document cannot answer questions"
            if status == JobStatus.FAILED
            else "the document is still being indexed"
        )
        raise ApiError(409, ErrorCode.CONFLICT, detail,
                       retryable=status != JobStatus.FAILED)

    semaphore: asyncio.Semaphore = request.app.state.qa_semaphore
    if semaphore.locked():
        raise ApiError(429, ErrorCode.RATE_LIMITED,
                       "QA concurrency limit reached; retry later", retryable=True)

    runtime = await _qa_runtime(request)
    scope = RetrievalScope(
        doc_id=doc_id,
        accepted_run_id=job["accepted_run_id"],
        index_fingerprint=job["index_fingerprint"],
    )

    deadline = Deadline(total_s=float(settings.qa_deadline_s))
    async with semaphore:
        try:
            async with asyncio.timeout(settings.qa_deadline_s):
                state = await run_qa(
                    runtime["graph"], question=body.question,
                    scope=scope, debug=body.debug, deadline=deadline,
                )
        except (TimeoutError, DeadlineExceeded) as exc:
            raise ApiError(
                504, ErrorCode.DEADLINE_EXCEEDED,
                f"QA deadline of {settings.qa_deadline_s}s exceeded", retryable=True,
            ) from exc
        except RetrievalError as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE,
                           f"{exc.code}: {exc.message}", retryable=True) from exc
        except ProviderUnavailable as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE, str(exc),
                           retryable=True) from exc

    outcome = state.get("outcome")
    if outcome == OUTCOME_ANSWERED:
        response = assemble_answer(state)
    elif outcome == OUTCOME_NOT_FOUND:
        response = AnswerResponse(answer=NO_ANSWER_MESSAGE, found=False, citations=[])
    else:  # OUTCOME_INVALID_OUTPUT
        raise ApiError(
            502, ErrorCode.MODEL_OUTPUT_INVALID,
            "the model failed to produce valid structured output after the "
            "allowed repair attempt", retryable=True,
        )

    if body.debug:
        response = response.model_copy(
            update={"diagnostics": _build_diagnostics(state, runtime, response)}
        )
    logger.info("question_answered", doc_id=doc_id, found=response.found,
                outcome=outcome, llm_calls=state.get("llm_calls", 0),
                repairs=state.get("repair_count", 0))
    return response


def _build_diagnostics(
    state: Any, runtime: dict[str, Any], response: AnswerResponse
) -> QaDiagnostics:
    retrieval = state.get("retrieval")
    diag = retrieval.diagnostics if retrieval is not None else None
    model_ids = dict(diag.model_ids) if diag else {}
    model_ids["llm"] = runtime["adapter"].capabilities.model
    return QaDiagnostics(
        dense_returned=diag.dense_returned if diag else None,
        sparse_returned=diag.sparse_returned if diag else None,
        fused_returned=diag.fused_returned if diag else 0,
        reranked_count=diag.reranked_count if diag else 0,
        context_count=diag.context_count if diag else 0,
        context_chunk_ids=list(diag.context_chunk_ids) if diag else [],
        # cited != context: only chunks actually backing citations in the answer
        cited_chunk_ids=sorted({c.chunk_id for c in response.citations}),
        rerank_scores=dict(diag.rerank_scores) if diag else {},
        stage_ms=dict(diag.stage_ms) if diag else {},
        model_ids=model_ids,
        token_usage={
            "input_tokens": state.get("usage_in", 0),
            "output_tokens": state.get("usage_out", 0),
            "llm_calls": state.get("llm_calls", 0),
        },
        branch_counts_source=diag.branch_counts_source if diag else None,
        diagnostic_warnings=list(diag.diagnostic_warnings) if diag else [],
    )
