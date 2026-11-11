"""Finite LangGraph QA workflow (§7.1-7.3). No checkpointer, no store, no
thread_id: a fresh state per ainvoke; the compiled graph object is shared, call
state never is. Transient infra failures raise OUT of the graph (503/504 at the
route); content failures route to the single allowed repair or a conservative
refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from app.config.settings import Settings
from app.inference.client import LocalModelsClient
from app.ingestion.chunker import LengthFn
from app.observability.logging import get_logger
from app.providers.factory import ChatAdapter, ModelOutputInvalid, ProviderUnavailable
from app.qa.grounding import LocatedQuote, ProvenanceError, validate_claims
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

logger = get_logger("qa.graph")

RECURSION_LIMIT = 16

OUTCOME_ANSWERED = "answered"
OUTCOME_NOT_FOUND = "not_found"
OUTCOME_INVALID_OUTPUT = "model_output_invalid"


class QaState(TypedDict, total=False):
    question: str
    debug: bool
    deadline: Any  # reliability.Deadline; monotonic budget shared by all stages
    scope: RetrievalScope
    retrieval: RetrievalResult | None
    short_to_chunk: dict[str, str]
    draft: DraftAnswer | None
    located: dict[int, list[LocatedQuote]] | None
    verification: Verification | None
    repair_count: int
    llm_calls: int
    repair_feedback: str | None
    usage_in: int
    usage_out: int
    outcome: str | None


@dataclass
class QaDependencies:
    settings: Settings
    qdrant: Any
    local_models: LocalModelsClient
    bm25: Bm25Adapter
    length_fn: LengthFn
    adapter: ChatAdapter


async def _llm_json_with_policy(
    state: QaState, deps: QaDependencies, *, system: str, user: str,
    json_schema: dict, max_tokens: int,
) -> tuple[dict, dict[str, int], int]:
    """Call the LLM with bounded transport retries (§8.6). EVERY transport attempt
    counts toward MAX_LLM_CALLS_PER_QUESTION (plan §10.3). Returns
    (raw, usage, attempts_used); raises ProviderUnavailable / ModelOutputInvalid /
    DeadlineExceeded."""
    import asyncio

    settings = deps.settings
    breaker = get_breaker(f"llm:{deps.adapter.capabilities.provider}")
    calls_used = state.get("llm_calls", 0)
    budget_left = settings.max_llm_calls_per_question - calls_used
    attempts_allowed = max(1, min(settings.llm_max_attempts, budget_left))
    deadline: Deadline | None = state.get("deadline")
    attempts = 0
    while True:
        attempts += 1
        stage_timeout = (
            deadline.stage_budget(settings.llm_read_timeout_s)
            if deadline is not None else settings.llm_read_timeout_s
        )
        try:
            breaker.before_call()
            async with asyncio.timeout(stage_timeout):
                raw, usage = await deps.adapter.generate_json(
                    system=system, user=user,
                    json_schema=json_schema, max_tokens=max_tokens,
                )
            breaker.record_success()
            return raw, usage, attempts
        except ModelOutputInvalid:
            breaker.record_success()  # the runtime responded; content is the issue
            raise
        except CircuitOpenError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except TimeoutError as exc:
            breaker.record_transient_failure()
            logger.warning("llm_outcome_uncertain", attempt=attempts,
                           note="stage deadline hit after send; provider may have "
                                "processed the request")
            if attempts >= attempts_allowed:
                raise ProviderUnavailable(
                    "LLM stage deadline exceeded", uncertain=True
                ) from exc
            await asyncio.sleep(backoff_delay(attempts - 1))
        except ProviderUnavailable as exc:
            breaker.record_transient_failure()
            if exc.uncertain:
                logger.warning("llm_outcome_uncertain", attempt=attempts,
                               error=str(exc)[:120])
            if attempts >= attempts_allowed:
                raise
            await asyncio.sleep(backoff_delay(attempts - 1))


def build_qa_graph(deps: QaDependencies) -> Any:
    settings = deps.settings
    max_repairs = settings.max_repair_attempts
    max_calls = settings.max_llm_calls_per_question

    async def retrieve(state: QaState) -> QaState:
        import asyncio

        deadline: Deadline | None = state.get("deadline")
        retrieval_budget = (
            deadline.stage_budget(300.0) if deadline is not None else 300.0
        )
        async with asyncio.timeout(retrieval_budget):
            result = await _do_retrieve(state)
        update: QaState = {"retrieval": result}
        if not result.context:
            update["outcome"] = OUTCOME_NOT_FOUND
        else:
            update["short_to_chunk"] = {
                f"source-{i + 1}": chunk.chunk_id
                for i, chunk in enumerate(result.context)
            }
        return update

    async def _do_retrieve(state: QaState) -> Any:
        return await hybrid_search(
            question=state["question"],
            scope=state["scope"],
            settings=settings,
            qdrant=deps.qdrant,
            local_models=deps.local_models,
            bm25=deps.bm25,
            length_fn=deps.length_fn,
            debug=state.get("debug", False),
            freeze_dir=(
                None
                if settings.freeze_candidates_dir is None
                else __import__("pathlib").Path(settings.freeze_candidates_dir)
            ),
        )

    def _can_repair(state: QaState) -> bool:
        return (
            state.get("repair_count", 0) < max_repairs
            and state.get("llm_calls", 0) + 1 <= max_calls
        )

    async def generate(state: QaState) -> QaState:
        if state.get("llm_calls", 0) + 1 > max_calls:
            return {"outcome": OUTCOME_INVALID_OUTPUT}
        retrieval = state["retrieval"]
        assert retrieval is not None
        chunk_to_short = {v: k for k, v in state["short_to_chunk"].items()}
        sources = render_sources(
            retrieval.context,
            {c.chunk_id: chunk_to_short[c.chunk_id] for c in retrieval.context},
        )
        user = generation_user_prompt(state["question"], sources)
        if state.get("repair_feedback"):
            user += (
                "\n\nПРЕДЫДУЩАЯ ПОПЫТКА ОТКЛОНЕНА СЕРВЕРОМ: "
                f"{state['repair_feedback']}\n"
                "Исправь ошибку. Используй только дословные цитаты из источников; "
                "если подтверждения нет — answerable=false."
            )
        max_tokens = (
            settings.llm_reasoning_budget_tokens + settings.llm_max_output_tokens
        )
        update: QaState = {
            "repair_feedback": None,
            "verification": None,
            "located": None,
        }
        try:
            raw, usage, attempts = await _llm_json_with_policy(
                state, deps, system=GENERATION_SYSTEM_V1, user=user,
                json_schema=DRAFT_JSON_SCHEMA, max_tokens=max_tokens,
            )
            update["llm_calls"] = state.get("llm_calls", 0) + attempts
            update["usage_in"] = state.get("usage_in", 0) + usage["input_tokens"]
            update["usage_out"] = state.get("usage_out", 0) + usage["output_tokens"]
            update["draft"] = DraftAnswer.model_validate(raw)
        except (ModelOutputInvalid, ValidationError) as exc:
            update["llm_calls"] = state.get("llm_calls", 0) + 1
            logger.warning("draft_invalid", error=str(exc)[:200])
            if _can_repair({**state, **update}):
                update["repair_count"] = state.get("repair_count", 0) + 1
                update["repair_feedback"] = f"невалидный JSON ответа: {str(exc)[:300]}"
                update["draft"] = None
            else:
                update["outcome"] = OUTCOME_INVALID_OUTPUT
        return update

    async def ground(state: QaState) -> QaState:
        draft = state["draft"]
        assert draft is not None
        if not draft.answerable or not draft.claims:
            return {"outcome": OUTCOME_NOT_FOUND}
        retrieval = state["retrieval"]
        assert retrieval is not None
        context_by_short = {
            short: next(c for c in retrieval.context if c.chunk_id == chunk_id)
            for short, chunk_id in state["short_to_chunk"].items()
        }
        try:
            located = validate_claims(draft.claims, context_by_short)
        except ProvenanceError as exc:
            logger.warning("provenance_rejected", error=str(exc)[:200])
            if _can_repair(state):
                return {
                    "repair_count": state.get("repair_count", 0) + 1,
                    "repair_feedback": f"проверка источников: {str(exc)[:300]}",
                    "draft": None,
                }
            return {"outcome": OUTCOME_NOT_FOUND}  # conservative refusal (§7.3)
        return {"located": located}

    async def verify(state: QaState) -> QaState:
        if state.get("llm_calls", 0) + 1 > max_calls:
            return {"outcome": OUTCOME_NOT_FOUND}  # cannot verify -> never emit unverified
        draft = state["draft"]
        located = state["located"]
        assert draft is not None and located is not None
        blocks = []
        for index, claim in enumerate(draft.claims):
            quotes = "\n".join(f'  ЦИТАТА: "{q.quote_raw}"' for q in located[index])
            blocks.append(f"CLAIM {index}: {claim.text}\n{quotes}")
        user = verifier_user_prompt(state["question"], "\n\n".join(blocks))
        update: QaState = {}
        try:
            raw, usage, attempts = await _llm_json_with_policy(
                state, deps, system=VERIFIER_SYSTEM_V1, user=user,
                json_schema=VERIFY_JSON_SCHEMA,
                max_tokens=settings.llm_reasoning_budget_tokens + 512,
            )
            update["llm_calls"] = state.get("llm_calls", 0) + attempts
            update["usage_in"] = state.get("usage_in", 0) + usage["input_tokens"]
            update["usage_out"] = state.get("usage_out", 0) + usage["output_tokens"]
            verification = Verification.model_validate(raw)
        except (ModelOutputInvalid, ValidationError) as exc:
            update["llm_calls"] = state.get("llm_calls", 0) + 1
            logger.warning("verifier_invalid", error=str(exc)[:200])
            update["outcome"] = OUTCOME_INVALID_OUTPUT
            return update
        update["verification"] = verification
        verdict_by_index = {v.claim_index: v.verdict for v in verification.verdicts}
        all_supported = all(
            verdict_by_index.get(i) == ClaimVerdict.SUPPORTED
            for i in range(len(draft.claims))
        )
        if all_supported and verification.coverage == Coverage.FULL:
            update["outcome"] = OUTCOME_ANSWERED
        elif _can_repair({**state, **update}):
            bad = [
                f"claim {i}: {verdict_by_index.get(i, 'нет вердикта')}"
                for i in range(len(draft.claims))
                if verdict_by_index.get(i) != ClaimVerdict.SUPPORTED
            ]
            if verification.coverage != Coverage.FULL:
                bad.append("покрытие вопроса неполное (partial)")
            update["repair_count"] = state.get("repair_count", 0) + 1
            update["repair_feedback"] = (
                "верификация отклонила ответ: " + "; ".join(bad)
            )
            update["draft"] = None
        else:
            update["outcome"] = OUTCOME_NOT_FOUND  # conservative refusal
        return update

    def route_after_retrieve(state: QaState) -> str:
        return END if state.get("outcome") else "generate"

    def route_after_generate(state: QaState) -> str:
        if state.get("outcome"):
            return END
        if state.get("draft") is None:
            return "generate"  # single repair loop (bounded by _can_repair)
        return "ground"

    def route_after_ground(state: QaState) -> str:
        if state.get("outcome"):
            return END
        if state.get("draft") is None:
            return "generate"
        return "verify"

    def route_after_verify(state: QaState) -> str:
        if state.get("outcome"):
            return END
        if state.get("draft") is None:
            return "generate"
        return END

    graph = StateGraph(QaState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("ground", ground)
    graph.add_node("verify", verify)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_conditional_edges("generate", route_after_generate)
    graph.add_conditional_edges("ground", route_after_ground)
    graph.add_conditional_edges("verify", route_after_verify)
    # §7.1: compile WITHOUT checkpointer/store — no persistence, no memory.
    return graph.compile()


async def run_qa(
    compiled_graph: Any, *, question: str, scope: RetrievalScope, debug: bool,
    deadline: Deadline | None = None,
) -> QaState:
    initial: QaState = {
        "question": question, "scope": scope, "debug": debug, "deadline": deadline,
        "repair_count": 0, "llm_calls": 0, "usage_in": 0, "usage_out": 0,
        "retrieval": None, "draft": None, "located": None, "verification": None,
        "repair_feedback": None, "outcome": None,
    }
    return await compiled_graph.ainvoke(
        initial, config={"recursion_limit": RECURSION_LIMIT}
    )
