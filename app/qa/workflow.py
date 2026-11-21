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
from app.qa.evidence import issue_evidence
from app.qa.extractive import (
    EXTRACT_VERIFIER,
    SELECTION_SYSTEM,
    bind_local_clauses,
    from_selection,
    reading_passages,
    retry_numeric_reference,
    selection_prompt,
    selection_schema,
)
from app.qa.grounding import (
    LocatedQuote,
    ProvenanceError,
    check_requested_value,
    qualify_numeric_claims,
    recover_evidence,
    remove_repeated_claims,
    restore_claim_wording,
    validate_claim_facts,
)
from app.qa.prompts import (
    GENERATION_SYSTEM_V1,
    SOURCE_PRECEDENCE_CHECK,
    VERIFIER_SYSTEM_V1,
    generation_user_prompt,
    render_sources,
    verifier_user_prompt,
)
from app.qa.relevance import explicit_source_priority, narrow_priority_evidence, refine_answer_scope
from app.qa.schemas import (
    DRAFT_JSON_SCHEMA,
    VERIFY_JSON_SCHEMA,
    ClaimVerdict,
    Coverage,
    DraftAnswer,
    Verification,
    is_incomplete_draft,
)
from app.qa.source_details import complete_source_details
from app.reliability.breaker import CircuitOpenError, get_breaker
from app.reliability.deadline import Deadline
from app.reliability.policy import backoff_delay
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import (
    RetrievalResult,
    RetrievalScope,
    effective_context_budget,
    hybrid_search,
)

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
    refusal_stage: str | None = None


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
    issues = [
        f"claim {v.claim_index} «{draft.claims[v.claim_index].text[:180]}»: "
        f"{v.verdict} {v.reason[:120]}".strip()
        for v in verification.verdicts if v.verdict != ClaimVerdict.SUPPORTED
    ]
    if verification.coverage != Coverage.FULL:
        issues.append("покрытие вопроса неполное: " + ", ".join(verification.coverage_gaps)[:460])
    return "; ".join(issues)


def _coverage_sources(spans: dict[str, LocatedQuote], located: dict[int, list[LocatedQuote]],
                      budget: int, length_fn: LengthFn) -> str:
    used = {(q.char_start, q.char_end) for group in located.values() for q in group}
    lines = []
    for key, span in spans.items():
        ref = (span.char_start, span.char_end)
        if ref in used:
            continue
        line = f"[{key}] {span.quote_raw}"
        size = length_fn(line)
        if size <= budget:
            lines.append(line)
            budget -= size
            used.add(ref)
    return "\n".join(lines)


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
        result.refusal_stage = "no_context"
        return result
    sources = {f"source-{i}": c for i, c in enumerate(retrieval.context, 1)}
    priority = explicit_source_priority(sources)
    spans = issue_evidence(sources)
    prompt = generation_user_prompt(question, render_sources(sources, spans))
    feedback = ""
    choices: dict[str, LocatedQuote] = {}
    for attempt in range(1 + settings.max_repair_attempts):
        if result.llm_calls >= settings.max_llm_calls_per_question:
            break
        result.repair_count = attempt
        stage = "generation"
        try:
            output_budget = settings.llm_reasoning_budget_tokens
            if choices:
                raw = await _call_json(
                    deps, result, deadline, system=SELECTION_SYSTEM,
                    user=selection_prompt(question, choices), schema=selection_schema(question),
                    max_tokens=output_budget + 384,
                )
                draft = from_selection(raw, choices, question)
            else:
                raw = await _call_json(
                    deps, result, deadline, system=GENERATION_SYSTEM_V1, user=prompt + feedback,
                    schema=DRAFT_JSON_SCHEMA,
                    max_tokens=output_budget + settings.llm_max_output_tokens,
                )
                draft = DraftAnswer.model_validate(raw)
            completed = complete_source_details(question, draft, spans)
            if completed == draft:
                bind_local_clauses(question, completed.claims, spans)
            draft = completed
            if not draft.answerable:
                logger.info("draft_not_answerable", repairs=attempt)
                result.refusal_stage = stage
                result.outcome = OUTCOME_NOT_FOUND
                if (attempt == 0 and settings.max_repair_attempts
                        and settings.max_llm_calls_per_question - result.llm_calls >= 2):
                    choices = reading_passages(question, spans)
                    if choices:
                        continue
                return result
            stage = "evidence"
            located = recover_evidence(draft.claims, sources, spans)
            if priority is not None:
                narrow_priority_evidence(question, draft.claims, located, sources, priority)
            stage = "facts"
            restore_claim_wording(draft.claims, located, sources, question,
                                  extractive=True, complete_qualifications=True)
            qualify_numeric_claims(draft.claims, located)
            located = refine_answer_scope(question, draft.claims, located, sources)
            located = remove_repeated_claims(draft.claims, located)
            validate_claim_facts(draft.claims, located, sources, question)
            check_requested_value(question, draft.claims)
            if result.llm_calls >= settings.max_llm_calls_per_question:
                result.refusal_stage = "verification_budget"
                result.outcome = OUTCOME_NOT_FOUND
                return result
            stage = "verification"
            blocks = []
            for index, claim in enumerate(draft.claims):
                quotes = "\n".join(
                    f'  ЦИТАТА ({q.chunk.scope_label or "раздел не указан"}): "{q.quote_raw}"'
                    for q in located[index]
                )
                blocks.append(f"CLAIM {index}: {claim.text}\n{quotes}")
            claims_block = "\n\n".join(blocks)
            budget = effective_context_budget(settings, deps.length_fn(question))
            remaining = min(1600, budget - deps.length_fn(claims_block))
            supplement = "" if choices else _coverage_sources(spans, located, remaining,
                                                               deps.length_fn)
            verifier = EXTRACT_VERIFIER if choices else VERIFIER_SYSTEM_V1
            if priority is not None:
                verifier += SOURCE_PRECEDENCE_CHECK
            raw = await _call_json(
                deps, result, deadline, system=verifier,
                user=verifier_user_prompt(question, claims_block, supplement),
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
            result.refusal_stage = None
            return result
        except (ModelOutputInvalid, ValidationError) as exc:
            result.outcome = (OUTCOME_NOT_FOUND if is_incomplete_draft(exc)
                              else OUTCOME_INVALID_OUTPUT)
            fields = ([{"path": list(issue["loc"]), "type": issue["type"]}
                       for issue in exc.errors(include_input=False, include_context=False)[:8]]
                      if isinstance(exc, ValidationError) else [])
            logger.warning("draft_invalid", kind=type(exc).__name__, stage=stage, fields=fields)
            result.refusal_stage = stage
            reason = "Выдай валидный JSON по схеме со всеми обязательными полями."
            if (attempt == 0 and stage == "generation" and settings.max_repair_attempts
                    and settings.max_llm_calls_per_question - result.llm_calls >= 2):
                choices = reading_passages(question, spans)
        except ProvenanceError as exc:
            result.outcome = OUTCOME_NOT_FOUND
            logger.warning("provenance_rejected", error=str(exc)[:200])
            result.refusal_stage = stage
            reason = str(exc)[:500]
            if attempt == 0 and stage == "facts" and retry_numeric_reference(question, reason):
                choices = reading_passages(question, spans)
        feedback = (
            f"\n\nПРЕДЫДУЩАЯ ПОПЫТКА ОТКЛОНЕНА СЕРВЕРОМ: {reason}\n"
            "Исправь указанный факт или выбери ID подтверждающих фрагментов. "
            "если подтверждения нет — answerable=false."
        )
    return result
