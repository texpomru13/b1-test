"""Stateless document QA with a bounded generation, grounding and verification loop."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.config.settings import Settings
from app.inference.client import LocalModelsClient
from app.ingestion.chunker import LengthFn
from app.observability.logging import get_logger
from app.providers.factory import ChatAdapter, ModelOutputInvalid, ProviderUnavailable
from app.qa.grounding import (
    LocatedQuote,
    ProvenanceError,
    check_requested_value,
    validate_claims,
)
from app.qa.prompts import (
    GENERATION_SYSTEM_V1,
    VERIFIER_SYSTEM_V1,
    generation_user_prompt,
    render_sources,
    verifier_user_prompt,
)
from app.qa.schemas import (
    DRAFT_JSON_SCHEMA,
    VERIFY_JSON_SCHEMA,
    ClaimVerdict,
    Coverage,
    DraftAnswer,
    Verification,
)
from app.reliability.breaker import CircuitOpenError, get_breaker
from app.reliability.deadline import Deadline
from app.reliability.policy import backoff_delay
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import RetrievalResult, RetrievalScope, hybrid_search

logger = get_logger("qa.workflow")
OUTCOME_ANSWERED = "answered"
OUTCOME_NOT_FOUND = "not_found"
OUTCOME_INVALID_OUTPUT = "model_output_invalid"


@dataclass
class QaDependencies:
    settings: Settings
    qdrant: Any
    local_models: LocalModelsClient
    bm25: Bm25Adapter
    length_fn: LengthFn
    adapter: ChatAdapter


@dataclass
class QaResult:
    retrieval: RetrievalResult
    outcome: str = OUTCOME_NOT_FOUND
    draft: DraftAnswer | None = None
    located: dict[int, list[LocatedQuote]] | None = None
    repair_count: int = 0
    llm_calls: int = 0
    usage_in: int = 0
    usage_out: int = 0


async def _call_json(
    deps: QaDependencies, result: QaResult, deadline: Deadline | None, *,
    system: str, user: str, schema: dict, max_tokens: int,
) -> dict:
    """Count every sent request, including retries followed by invalid model output."""
    settings = deps.settings
    breaker = get_breaker(f"llm:{deps.adapter.capabilities.provider}")
    remaining = settings.max_llm_calls_per_question - result.llm_calls
    attempts = min(settings.llm_max_attempts, remaining)
    for attempt in range(attempts):
        timeout = settings.llm_read_timeout_s
        budget = deadline.stage_budget(timeout) if deadline else timeout
        try:
            breaker.before_call()
        except CircuitOpenError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        result.llm_calls += 1
        try:
            async with asyncio.timeout(budget):
                raw, usage = await deps.adapter.generate_json(
                    system=system, user=user, json_schema=schema, max_tokens=max_tokens,
                )
            breaker.record_success()
            result.usage_in += usage["input_tokens"]
            result.usage_out += usage["output_tokens"]
            return raw
        except ModelOutputInvalid:
            breaker.record_success()
            raise
        except asyncio.CancelledError:
            # Release a half-open probe when the request's outer deadline cancels it.
            breaker.record_config_failure()
            raise
        except (TimeoutError, ProviderUnavailable) as exc:
            error = (ProviderUnavailable("LLM stage deadline exceeded", uncertain=True)
                     if isinstance(exc, TimeoutError) else exc)
            if not error.retryable:
                breaker.record_config_failure()
                raise
            breaker.record_transient_failure()
            if error.uncertain:
                logger.warning("llm_outcome_uncertain", attempt=attempt + 1)
            if attempt + 1 == attempts:
                if error is exc:
                    raise
                raise error from exc
            await asyncio.sleep(error.retry_after_s or backoff_delay(attempt))
    raise RuntimeError("LLM call attempted with no remaining budget")


def _verdict_feedback(draft: DraftAnswer, verification: Verification) -> str:
    expected = set(range(len(draft.claims)))
    verdicts = {v.claim_index: v.verdict for v in verification.verdicts}
    if set(verdicts) != expected or len(verification.verdicts) != len(expected):
        raise ModelOutputInvalid("verifier must return one verdict per claim")
    issues = [f"claim {i}: {v}" for i, v in verdicts.items() if v != ClaimVerdict.SUPPORTED]
    if verification.coverage != Coverage.FULL:
        issues.append("покрытие вопроса неполное (partial)")
    return "; ".join(issues)


async def run_qa(
    deps: QaDependencies, *, question: str, scope: RetrievalScope, debug: bool,
    deadline: Deadline | None = None,
) -> QaResult:
    settings = deps.settings
    async with asyncio.timeout(deadline.stage_budget(300.0) if deadline else 300.0):
        retrieval = await hybrid_search(
            question=question, scope=scope, settings=settings, qdrant=deps.qdrant,
            local_models=deps.local_models, bm25=deps.bm25, length_fn=deps.length_fn,
            debug=debug,
        )
    result = QaResult(retrieval)
    if not retrieval.context:
        return result
    sources = {f"source-{i}": c for i, c in enumerate(retrieval.context, 1)}
    prompt = generation_user_prompt(question, render_sources(sources))
    feedback = ""
    for attempt in range(1 + settings.max_repair_attempts):
        if result.llm_calls >= settings.max_llm_calls_per_question:
            break
        result.repair_count = attempt
        try:
            raw = await _call_json(
                deps, result, deadline, system=GENERATION_SYSTEM_V1, user=prompt + feedback,
                schema=DRAFT_JSON_SCHEMA,
                max_tokens=settings.llm_reasoning_budget_tokens + settings.llm_max_output_tokens,
            )
            draft = DraftAnswer.model_validate(raw)
            if not draft.answerable:
                logger.info("draft_not_answerable", repairs=attempt)
                result.outcome = OUTCOME_NOT_FOUND
                return result
            located = validate_claims(draft.claims, sources)
            check_requested_value(question, draft.claims)
            if result.llm_calls >= settings.max_llm_calls_per_question:
                result.outcome = OUTCOME_NOT_FOUND
                return result
            blocks = []
            for index, claim in enumerate(draft.claims):
                quotes = "\n".join(f'  ЦИТАТА: "{q.quote_raw}"' for q in located[index])
                blocks.append(f"CLAIM {index}: {claim.text}\n{quotes}")
            raw = await _call_json(
                deps, result, deadline, system=VERIFIER_SYSTEM_V1,
                user=verifier_user_prompt(question, "\n\n".join(blocks)),
                schema=VERIFY_JSON_SCHEMA, max_tokens=settings.llm_reasoning_budget_tokens + 512,
            )
            verification = Verification.model_validate(raw)
            issues = _verdict_feedback(draft, verification)
            logger.info("verifier_verdict", coverage=verification.coverage,
                        verdicts=[v.verdict for v in verification.verdicts], repairs=attempt)
            if issues:
                raise ProvenanceError(issues)
            result.draft, result.located = draft, located
            result.outcome = OUTCOME_ANSWERED
            return result
        except (ModelOutputInvalid, ValidationError) as exc:
            result.outcome = OUTCOME_INVALID_OUTPUT
            logger.warning("draft_invalid", kind=type(exc).__name__)
            reason = "Выдай валидный JSON по схеме со всеми обязательными полями."
        except ProvenanceError as exc:
            result.outcome = OUTCOME_NOT_FOUND
            logger.warning("provenance_rejected", error=str(exc)[:200])
            reason = str(exc)[:300]
        feedback = (
            f"\n\nПРЕДЫДУЩАЯ ПОПЫТКА ОТКЛОНЕНА СЕРВЕРОМ: {reason}\n"
            "Исправь ошибку. Используй только дословные цитаты из источников; "
            "если подтверждения нет — answerable=false."
        )
    return result
