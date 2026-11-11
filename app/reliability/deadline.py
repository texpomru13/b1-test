"""Monotonic request deadline (§8.6): every stage receives the REMAINING budget,
so per-stage timeouts can never add up beyond the request deadline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class DeadlineExceeded(Exception):
    pass


@dataclass
class Deadline:
    total_s: float
    _started: float = field(default_factory=time.monotonic)

    def remaining(self) -> float:
        return self.total_s - (time.monotonic() - self._started)

    def stage_budget(self, stage_timeout_s: float) -> float:
        """min(stage timeout, remaining); raises when the budget is exhausted."""
        remaining = self.remaining()
        if remaining <= 0:
            raise DeadlineExceeded(f"deadline of {self.total_s}s exhausted")
        return min(stage_timeout_s, remaining)
