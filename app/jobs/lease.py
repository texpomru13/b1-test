"""Job lease ownership with a heartbeat (ADR 003 §5).

A worker owns a job from the moment it starts executing it until it finishes:
the lease is acquired BEFORE the job becomes `running` and a background
heartbeat keeps it alive through every long step (DOCX conversion, parsing,
splitting, a slow embedding batch). The reconciler treats `running` without a
lease as a crashed worker, so a live worker must never be lease-less.

If the worker process dies, the heartbeat dies with it and the lease expires
after LEASE_TTL_MS — that expiry is the crash signal. A hung task cannot keep
the lease forever: Celery's hard time limit kills the process.
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Protocol

from app.observability.logging import get_logger

logger = get_logger("jobs.lease")

LEASE_TTL_MS = 60_000
# Several renewals fit into one TTL, so a single missed beat is harmless.
HEARTBEAT_INTERVAL_S = 15.0


class _LeaseRenewer(Protocol):
    def renew_lease(self, job_id: str, token: str, *, ttl_ms: int) -> bool: ...


class LeaseKeeper:
    """Renews an already acquired lease on a background thread until stopped.

    ``lost`` becomes true when the renewal reports that the token no longer owns
    the lease (it expired and/or another worker took the job over). The owner
    must then step aside without touching the job state.
    """

    def __init__(
        self,
        lifecycle: _LeaseRenewer,
        job_id: str,
        token: str,
        *,
        ttl_ms: int = LEASE_TTL_MS,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._lifecycle = lifecycle
        self._job_id = job_id
        self._token = token
        self._ttl_ms = ttl_ms
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(
            target=self._beat, name=f"lease-heartbeat-{job_id[:8]}", daemon=True
        )

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def __enter__(self) -> LeaseKeeper:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _beat(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                owned = self._lifecycle.renew_lease(
                    self._job_id, self._token, ttl_ms=self._ttl_ms
                )
            except Exception as exc:
                # Redis hiccup: keep trying. If it outlasts the TTL the lease
                # expires and the next successful call reports the loss.
                logger.warning("lease_heartbeat_failed", job_id=self._job_id,
                               error=type(exc).__name__)
                continue
            if not owned:
                self._lost.set()
                logger.warning("lease_lost", job_id=self._job_id)
                return
