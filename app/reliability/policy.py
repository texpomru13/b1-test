"""Bounded retry with exponential backoff + full jitter (§8.6).

Single-owner rule: ONLY this layer retries. SDK retries are disabled
(max_retries=0 on every LLM client), Celery tasks never self-retry (attempts are
counted in Redis), qdrant-client/httpx perform no implicit retries.

Classification:
- qdrant-client errors are unwrapped first: ResponseHandlingException -> its
  underlying transport error, UnexpectedResponse -> its HTTP status.
- RETRY: connect/network errors, HTTP 429 (honoring Retry-After), 502/503/504.
- NEVER: 400/401/403/404/409/422 and schema/validation errors.
- Read timeouts AFTER a request was sent are retried within the bound but logged
  as an UNCERTAIN outcome: the provider may have processed (and billed) the call.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app.observability.logging import get_logger

logger = get_logger("reliability.retry")


NON_RETRYABLE_STATUS = {400, 401, 403, 404, 409, 413, 415, 422}
RETRYABLE_STATUS = {429, 502, 503, 504}


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    uncertain: bool = False  # request may have been processed by the provider
    retry_after_s: float | None = None


def _status_decision(status: int | None, retry_after_raw: str | None) -> RetryDecision:
    if status in RETRYABLE_STATUS:
        retry_after = None
        if retry_after_raw is not None:
            try:
                retry_after = max(0.0, float(retry_after_raw))
            except ValueError:
                retry_after = None
        return RetryDecision(retry=True, retry_after_s=retry_after)
    return RetryDecision(retry=False)


def _classify_qdrant(exc: BaseException) -> RetryDecision | None:
    """qdrant-client wraps transport failures and non-2xx answers in its own
    exception types; without unwrapping them every Qdrant outage would look
    non-retryable. Returns None when ``exc`` is not a qdrant-client error."""
    try:
        from qdrant_client.http.exceptions import (
            ResponseHandlingException,
            UnexpectedResponse,
        )
    except ImportError:  # pragma: no cover - qdrant-client is a hard dependency
        return None
    if isinstance(exc, ResponseHandlingException):
        # .source is the underlying httpx error (connect/timeout/protocol) or a
        # response-parsing error; classify the cause itself.
        return classify_exception(exc.source)
    if isinstance(exc, UnexpectedResponse):
        headers = exc.headers
        return _status_decision(
            exc.status_code, headers.get("Retry-After") if headers is not None else None
        )
    return None


def classify_exception(exc: BaseException) -> RetryDecision:
    qdrant_decision = _classify_qdrant(exc)
    if qdrant_decision is not None:
        return qdrant_decision
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in NON_RETRYABLE_STATUS:
            return RetryDecision(retry=False)
        if status in RETRYABLE_STATUS:
            retry_after = None
            raw = exc.response.headers.get("Retry-After")
            if raw is not None:
                try:
                    retry_after = max(0.0, float(raw))
                except ValueError:
                    retry_after = None
            return RetryDecision(retry=True, retry_after_s=retry_after)
        return RetryDecision(retry=False)
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return RetryDecision(retry=True)  # nothing was sent: certain, safe retry
    if isinstance(exc, httpx.ReadTimeout | httpx.WriteError | httpx.RemoteProtocolError):
        return RetryDecision(retry=True, uncertain=True)  # may have been processed
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return RetryDecision(retry=True, uncertain=True)
    if isinstance(exc, ConnectionError | OSError):
        return RetryDecision(retry=True)
    return RetryDecision(retry=False)


def backoff_delay(attempt: int, *, base_s: float = 0.5, cap_s: float = 10.0) -> float:
    """Full jitter: uniform(0, min(cap, base * 2^attempt))."""
    return random.uniform(0.0, min(cap_s, base_s * (2 ** attempt)))


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    op_name: str,
    max_attempts: int,
    base_s: float = 0.5,
    cap_s: float = 10.0,
    classify: Callable[[BaseException], RetryDecision] = classify_exception,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    do_sleep = sleep or asyncio.sleep
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await operation()
        except BaseException as exc:
            decision = classify(exc)
            last = exc
            if decision.uncertain:
                logger.warning(
                    "llm_outcome_uncertain" if "llm" in op_name else "outcome_uncertain",
                    op=op_name, attempt=attempt + 1,
                    error=type(exc).__name__,
                    note="request may have been processed/billed by the provider",
                )
            if not decision.retry or attempt + 1 >= max_attempts:
                raise
            delay = (
                decision.retry_after_s
                if decision.retry_after_s is not None
                else backoff_delay(attempt, base_s=base_s, cap_s=cap_s)
            )
            logger.info("retrying", op=op_name, attempt=attempt + 1, delay_s=round(delay, 2))
            await do_sleep(delay)
    raise last if last else RuntimeError("retry loop exited without result")


def retry_sync[T](
    operation: Callable[[], T],
    *,
    op_name: str,
    max_attempts: int,
    base_s: float = 0.5,
    cap_s: float = 10.0,
    classify: Callable[[BaseException], RetryDecision] = classify_exception,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return operation()
        except BaseException as exc:
            decision = classify(exc)
            last = exc
            if decision.uncertain:
                logger.warning("outcome_uncertain", op=op_name, attempt=attempt + 1,
                               error=type(exc).__name__)
            if not decision.retry or attempt + 1 >= max_attempts:
                raise
            delay = (
                decision.retry_after_s
                if decision.retry_after_s is not None
                else backoff_delay(attempt, base_s=base_s, cap_s=cap_s)
            )
            logger.info("retrying", op=op_name, attempt=attempt + 1, delay_s=round(delay, 2))
            sleep(delay)
    raise last if last else RuntimeError("retry loop exited without result")
