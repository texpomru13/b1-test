import asyncio

import httpx
import pytest
from app.reliability.breaker import CircuitBreaker, CircuitOpenError
from app.reliability.deadline import Deadline, DeadlineExceeded
from app.reliability.policy import classify_exception, retry_async


def _status_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(status, request=request, headers=headers or {})
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_classification_rules() -> None:
    assert classify_exception(_status_error(401)).retry is False  # never retried
    assert classify_exception(_status_error(403)).retry is False
    assert classify_exception(_status_error(422)).retry is False
    d429 = classify_exception(_status_error(429, {"Retry-After": "7"}))
    assert d429.retry is True and d429.retry_after_s == 7.0
    assert classify_exception(_status_error(503)).retry is True
    connect = classify_exception(httpx.ConnectError("x"))
    assert connect.retry is True and connect.uncertain is False  # nothing was sent
    read_timeout = classify_exception(httpx.ReadTimeout("x"))
    assert read_timeout.retry is True and read_timeout.uncertain is True  # maybe billed


async def test_401_single_attempt_429_honors_retry_after() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def op_401() -> None:
        calls["n"] += 1
        raise _status_error(401)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(op_401, op_name="t", max_attempts=3, sleep=fake_sleep)
    assert calls["n"] == 1  # 401 is NEVER retried
    assert sleeps == []

    calls["n"] = 0

    async def op_429_then_ok() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(429, {"Retry-After": "3"})
        return "ok"

    assert await retry_async(op_429_then_ok, op_name="t", max_attempts=3,
                             sleep=fake_sleep) == "ok"
    assert calls["n"] == 2
    assert sleeps == [3.0]  # Retry-After honored exactly


async def test_attempts_bounded() -> None:
    calls = {"n": 0}

    async def always_down() -> None:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(httpx.ConnectError):
        await retry_async(always_down, op_name="t", max_attempts=3, sleep=no_sleep)
    assert calls["n"] == 3


def test_breaker_open_halfopen_close_cycle() -> None:
    breaker = CircuitBreaker("t", failure_threshold=3, open_seconds=30.0)
    for _ in range(3):
        breaker.before_call()
        breaker.record_transient_failure()
    with pytest.raises(CircuitOpenError):
        breaker.before_call()  # open
    breaker._opened_at -= 31.0  # type: ignore[operator]  # fast-forward time
    breaker.before_call()  # half-open: this call is the single probe
    with pytest.raises(CircuitOpenError):
        breaker.before_call()  # concurrent probe rejected
    breaker.record_success()
    breaker.before_call()  # closed again


def test_breaker_config_errors_do_not_trip() -> None:
    breaker = CircuitBreaker("t", failure_threshold=2)
    for _ in range(10):
        breaker.before_call()
        breaker.record_config_failure()  # 4xx-style: never opens
    breaker.before_call()


def test_deadline_stage_budget() -> None:
    deadline = Deadline(total_s=100.0)
    assert deadline.stage_budget(30.0) == 30.0  # stage timeout smaller
    assert deadline.stage_budget(1000.0) < 100.0  # capped by remaining
    exhausted = Deadline(total_s=0.0)
    with pytest.raises(DeadlineExceeded):
        exhausted.stage_budget(10.0)


async def test_graph_counts_transport_retries_in_llm_budget() -> None:
    from app.providers.factory import ProviderUnavailable
    from app.qa.graph import OUTCOME_ANSWERED, build_qa_graph, run_qa
    from app.retrieval.search import RetrievalScope

    from tests.unit.test_qa_graph import (
        VERDICT_OK,
        ScriptedAdapter,
        good_draft,
        make_deps,
    )

    quote = "уплачивает неустойку 0,1% в день, но не более 10% цены договора"
    adapter = ScriptedAdapter([
        ProviderUnavailable("blip", uncertain=True),  # transport attempt 1 (counted)
        good_draft(quote),                            # attempt 2 -> draft ok
        VERDICT_OK,                                   # verifier
    ])
    graph = build_qa_graph(make_deps(adapter))
    state = await run_qa(graph, question="q",
                         scope=RetrievalScope("doc", "run", "fp"), debug=False)
    assert state["outcome"] == OUTCOME_ANSWERED
    # 2 transport attempts for generation + 1 verifier call = 3 of the budget of 4
    assert state["llm_calls"] == 3


async def test_graph_raises_provider_unavailable_when_llm_down() -> None:
    from app.providers.factory import ProviderUnavailable
    from app.qa.graph import build_qa_graph, run_qa
    from app.retrieval.search import RetrievalScope

    from tests.unit.test_qa_graph import ScriptedAdapter, make_deps

    adapter = ScriptedAdapter([
        ProviderUnavailable("down"), ProviderUnavailable("down"),
        ProviderUnavailable("down"), ProviderUnavailable("down"),
    ])
    graph = build_qa_graph(make_deps(adapter))
    with pytest.raises(ProviderUnavailable):  # route maps this to 503, never 200
        await run_qa(graph, question="q",
                     scope=RetrievalScope("doc", "run", "fp"), debug=False)
    assert len(adapter.prompts) == 2  # bounded by LLM_MAX_ATTEMPTS, not unbounded


async def test_sleepless() -> None:
    # keep asyncio import used explicitly (backoff sleeps are real but tiny in tests)
    await asyncio.sleep(0)


def test_qdrant_client_errors_are_unwrapped() -> None:
    """qdrant-client wraps transport errors and non-2xx answers in its own types;
    a temporary Qdrant outage must classify as retryable, a bad request must not."""
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )

    refused = classify_exception(ResponseHandlingException(httpx.ConnectError("refused")))
    assert refused.retry is True and refused.uncertain is False
    timed_out = classify_exception(ResponseHandlingException(httpx.ReadTimeout("slow")))
    assert timed_out.retry is True and timed_out.uncertain is True
    # a response-parsing failure is not transient
    assert classify_exception(ResponseHandlingException(ValueError("bad json"))).retry is False

    def unexpected(status: int, headers: dict | None = None) -> UnexpectedResponse:
        return UnexpectedResponse(
            status_code=status, reason_phrase="", content=b"",
            headers=httpx.Headers(headers or {}),
        )

    assert classify_exception(unexpected(503)).retry is True
    assert classify_exception(unexpected(502)).retry is True
    throttled = classify_exception(unexpected(429, {"Retry-After": "3"}))
    assert throttled.retry is True and throttled.retry_after_s == 3.0
    assert classify_exception(unexpected(400)).retry is False
    assert classify_exception(unexpected(404)).retry is False
