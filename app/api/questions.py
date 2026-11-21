"""Question endpoint. Each request uses the published index of one document."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Request

from app.api.errors import api_error_responses
from app.domain.job import JobStatus
from app.ingestion.chunker import ApproxTokenLength, HFTokenizerLength
from app.providers.factory import (
    ProviderConfigError,
    ProviderUnavailable,
    build_chat_adapter,
)
from app.qa.assemble import assemble_answer
from app.qa.workflow import (
    OUTCOME_ANSWERED,
    OUTCOME_NOT_FOUND,
    QaDependencies,
    QaResult,
    run_qa,
)
from app.reliability.deadline import Deadline, DeadlineExceeded
from app.retrieval.search import RetrievalError, RetrievalScope
from app.schemas.errors import ApiError, ErrorCode
from app.schemas.qa import (
    NO_ANSWER_MESSAGE,
    AnswerResponse,
    QaDiagnostics,
    QuestionRequest,
)
from app.storage.qdrant_repo import FingerprintMismatchError

router = APIRouter(prefix="/api/v1", tags=["questions"])
logger = structlog.get_logger("api.questions")


async def _qa_runtime(request: Request) -> QaDependencies:
    """Share adapters and tokenizer across questions."""
    app = request.app
    async with app.state.qa_init_lock:
        if app.state.qa_runtime is not None:
            return app.state.qa_runtime
        settings = app.state.settings
        try:
            adapter = build_chat_adapter(settings)
        except ProviderConfigError as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE, str(exc),
                           retryable=False) from exc
        length_fn: Any
        if settings.embedding_tokenizer_path:
            length_fn = HFTokenizerLength(
                Path(settings.embedding_tokenizer_path)
            )
        else:
            length_fn = ApproxTokenLength()
        app.state.qa_runtime = QaDependencies(
            settings=settings,
            qdrant=app.state.qdrant_repo._client,
            local_models=app.state.local_models,
            bm25=app.state.bm25,
            length_fn=length_fn,
            adapter=adapter,
        )
    return app.state.qa_runtime


# exclude_UNSET (not exclude_none): `diagnostics` is omitted unless debug=true, but
# nullable required fields (citation `section`, diagnostics branch counts) are kept
# as explicit nulls so every 200 body validates against the published schema.
@router.post("/documents/{doc_id}/questions", response_model=AnswerResponse,
             response_model_exclude_unset=True,
             responses=api_error_responses(404, 409, 422, 429, 502, 503, 504))
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
    if status in {JobStatus.FAILED, JobStatus.COMPENSATING, JobStatus.CLEANUP_PENDING}:
        raise ApiError(409, ErrorCode.CONFLICT, "indexing failed; check the indexing job")
    if status != JobStatus.SUCCEEDED:
        raise ApiError(409, ErrorCode.CONFLICT, "the document is still being indexed",
                       retryable=True)
    if not job.get("accepted_run_id"):
        raise ApiError(409, ErrorCode.CONFLICT, "the document has no published index")

    semaphore: asyncio.Semaphore = request.app.state.qa_semaphore
    if semaphore.locked():
        raise ApiError(429, ErrorCode.RATE_LIMITED,
                       "QA concurrency limit reached; retry later", retryable=True)

    scope = RetrievalScope(
        doc_id=doc_id,
        accepted_run_id=job["accepted_run_id"],
        index_fingerprint=job["index_fingerprint"],
    )

    deadline = Deadline(total_s=float(settings.qa_deadline_s))
    async with semaphore:
        try:
            async with asyncio.timeout(settings.qa_deadline_s):
                runtime = await _qa_runtime(request)
                state = await run_qa(
                    runtime, question=body.question,
                    scope=scope, debug=body.debug, deadline=deadline,
                )
        except (TimeoutError, DeadlineExceeded) as exc:
            raise ApiError(
                504, ErrorCode.DEADLINE_EXCEEDED,
                f"QA deadline of {settings.qa_deadline_s}s exceeded", retryable=True,
            ) from exc
        except FingerprintMismatchError as exc:
            raise ApiError(409, ErrorCode.CONFLICT, str(exc), retryable=False) from exc
        except RetrievalError as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE,
                           f"{exc.code}: {exc.message}", retryable=exc.retryable) from exc
        except ProviderUnavailable as exc:
            raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE, str(exc),
                           retryable=exc.retryable) from exc

    outcome = state.outcome
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
                outcome=outcome, llm_calls=state.llm_calls,
                repairs=state.repair_count)
    return response


def _build_diagnostics(
    state: QaResult, runtime: QaDependencies, response: AnswerResponse
) -> QaDiagnostics:
    diag = state.retrieval.diagnostics
    model_ids = dict(diag.model_ids)
    model_ids["llm"] = runtime.adapter.capabilities.model
    return QaDiagnostics(
        dense_returned=diag.dense_returned,
        sparse_returned=diag.sparse_returned,
        fused_returned=diag.fused_returned,
        reranked_count=diag.reranked_count,
        context_count=diag.context_count,
        context_chunk_ids=list(diag.context_chunk_ids),
        # cited != context: only chunks actually backing citations in the answer
        cited_chunk_ids=sorted({c.chunk_id for c in response.citations}),
        rerank_scores=dict(diag.rerank_scores),
        stage_ms=dict(diag.stage_ms),
        model_ids=model_ids,
        token_usage={
            "input_tokens": state.usage_in,
            "output_tokens": state.usage_out,
            "llm_calls": state.llm_calls,
        },
        branch_counts_source=diag.branch_counts_source,
        diagnostic_warnings=list(diag.diagnostic_warnings),
        refusal_stage=state.refusal_stage,
    )
