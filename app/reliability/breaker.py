"""Process-local circuit breaker (§8.6): 5 consecutive transient failures open
the circuit for 30s; one half-open probe; configuration errors (4xx) never count.
One API worker runs per deployment, so process-local state is sufficient (the
multi-replica story is an explicit production-backlog item).
"""

from __future__ import annotations

import threading
import time

from app.observability.logging import get_logger

logger = get_logger("reliability.breaker")


class CircuitOpenError(Exception):
    def __init__(self, name: str, retry_in_s: float) -> None:
        super().__init__(
            f"circuit '{name}' is open; retry in {retry_in_s:.0f}s"
        )
        self.retry_in_s = retry_in_s


class CircuitBreaker:
    def __init__(
        self, name: str, *, failure_threshold: int = 5, open_seconds: float = 30.0
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._open_s = open_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_inflight = False
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self._open_s:
                raise CircuitOpenError(self.name, self._open_s - elapsed)
            if self._half_open_probe_inflight:
                raise CircuitOpenError(self.name, 1.0)  # a probe is already running
            self._half_open_probe_inflight = True  # this call IS the probe

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:
                logger.info("circuit_closed", name=self.name)
            self._failures = 0
            self._opened_at = None
            self._half_open_probe_inflight = False

    def record_transient_failure(self) -> None:
        with self._lock:
            self._half_open_probe_inflight = False
            if self._opened_at is not None:
                self._opened_at = time.monotonic()  # failed probe: stay open
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()
                logger.warning("circuit_opened", name=self.name,
                               failures=self._failures)

    def record_config_failure(self) -> None:
        with self._lock:  # 4xx-style errors do not trip the breaker
            self._half_open_probe_inflight = False


_registry: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(name: str) -> CircuitBreaker:
    with _registry_lock:
        if name not in _registry:
            _registry[name] = CircuitBreaker(name)
        return _registry[name]
