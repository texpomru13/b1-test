"""Sync job lifecycle operations for the Celery worker and the dispatcher process."""

from __future__ import annotations

from typing import Any

import redis

from app.storage import keys
from app.storage.redis_repo import utc_now_iso


def create_sync_client(url: str, *, socket_timeout: float | None = None) -> redis.Redis:
    # For blocking commands (BLMOVE) socket_timeout must exceed the blocking
    # timeout, otherwise the client raises TimeoutError instead of returning None.
    return redis.Redis.from_url(url, decode_responses=True, socket_timeout=socket_timeout)


class TransitionResult:
    def __init__(self, verdict: str, previous_status: str) -> None:
        self.verdict = verdict  # OK | INVALID_STATE | NOT_FOUND
        self.previous_status = previous_status

    @property
    def ok(self) -> bool:
        return self.verdict == "OK"


class SyncJobLifecycle:
    def __init__(self, client: redis.Redis) -> None:
        self._client = client
        self._transition = client.register_script(keys.TRANSITION_LUA)
        self._publish = client.register_script(keys.PUBLISH_LUA)
        self._renew = client.register_script(keys.RENEW_LUA)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        data = self._client.hgetall(keys.job_key(job_id))
        if not data:
            return None
        return {str(k): str(v) for k, v in data.items()}

    def transition(
        self,
        job_id: str,
        *,
        expected: list[str],
        new_status: str,
        extra: dict[str, str] | None = None,
    ) -> TransitionResult:
        flat: list[str] = []
        for key, value in (extra or {}).items():
            flat.extend([key, value])
        raw = self._transition(
            keys=[keys.job_key(job_id)],
            args=[",".join(expected), new_status, utc_now_iso(), *flat],
        )
        verdict, previous = (str(part) for part in raw)
        return TransitionResult(verdict, previous)

    def set_fields(self, job_id: str, fields: dict[str, str]) -> None:
        payload = {**fields, "updated_at": utc_now_iso()}
        self._client.hset(keys.job_key(job_id), mapping=payload)  # type: ignore[arg-type]

    def incr_attempt(self, job_id: str) -> int:
        return int(self._client.hincrby(keys.job_key(job_id), "attempt", 1))

    def mark_dispatched(self, job_id: str) -> None:
        self._client.hset(keys.job_key(job_id), "dispatched", "1")

    def is_doc_referenced(self, doc_id: str) -> bool:
        return bool(self._client.exists(keys.doc_key(doc_id)))

    # --- lease + publication (ADR 003) ---

    def acquire_lease(self, job_id: str, *, ttl_ms: int) -> str | None:
        import uuid as _uuid

        token = _uuid.uuid4().hex
        acquired = self._client.set(keys.lease_key(job_id), token, nx=True, px=ttl_ms)
        return token if acquired else None

    def renew_lease(self, job_id: str, token: str, *, ttl_ms: int) -> bool:
        return bool(self._renew(keys=[keys.lease_key(job_id)], args=[token, str(ttl_ms)]))

    def release_lease(self, job_id: str, token: str) -> None:
        # Best-effort owner-only release (compare-and-delete via renewal to 1ms).
        self._renew(keys=[keys.lease_key(job_id)], args=[token, "1"])

    def publish(self, job_id: str, *, token: str, run_id: str) -> TransitionResult:
        raw = self._publish(
            keys=[keys.lease_key(job_id), keys.job_key(job_id)],
            args=[token, run_id, utc_now_iso()],
        )
        verdict, detail = (str(part) for part in raw)
        return TransitionResult(verdict, detail)
