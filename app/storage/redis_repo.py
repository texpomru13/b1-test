"""Async Redis job lifecycle repository (API side).

Atomic submit (job + idempotency + outbox intent) and reads. State transitions on
the worker/dispatcher side live in `job_lifecycle_sync.py`; both share `keys.py`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from app.storage import keys


def create_redis_client(url: str) -> aioredis.Redis:
    return aioredis.from_url(url, decode_responses=True)


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class SubmitResult:
    def __init__(self, verdict: str, job_id: str, doc_id: str) -> None:
        self.verdict = verdict  # CREATED | DUPLICATE | CONFLICT
        self.job_id = job_id
        self.doc_id = doc_id


class RedisJobRepository:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client
        self._submit = client.register_script(keys.SUBMIT_LUA)

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def submit(
        self,
        *,
        idem_key: str,
        fingerprint: str,
        job_id: str,
        doc_id: str,
        job_fields: dict[str, str],
    ) -> SubmitResult:
        now_iso = utc_now_iso()
        fields: dict[str, str] = {
            "job_id": job_id,
            "doc_id": doc_id,
            "status": "accepted",
            "stage": "upload",
            "attempt": "0",
            "dispatched": "0",
            "created_at": now_iso,
            "updated_at": now_iso,
            **job_fields,
        }
        flat: list[str] = []
        for key, value in fields.items():
            flat.extend([key, value])
        raw = await self._submit(
            keys=[
                keys.idempotency_key(idem_key),
                keys.job_key(job_id),
                keys.outbox_pending_key(),
                keys.jobs_by_created_key(),
                keys.doc_key(doc_id),
            ],
            args=[fingerprint, job_id, doc_id, str(time.time()), *flat],
        )
        verdict, existing_job, existing_doc = (str(part) for part in raw)
        return SubmitResult(verdict, existing_job, existing_doc)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        data = await self._client.hgetall(keys.job_key(job_id))
        return {str(k): str(v) for k, v in data.items()} if data else None

    async def get_job_id_for_doc(self, doc_id: str) -> str | None:
        value = await self._client.get(keys.doc_key(doc_id))
        return str(value) if value else None

    async def list_jobs(self, *, offset: int, limit: int) -> tuple[list[dict[str, Any]], bool]:
        """Newest-first page of jobs; returns (items, has_more)."""
        raw_ids = await self._client.zrevrange(
            keys.jobs_by_created_key(), offset, offset + limit
        )
        job_ids = [str(job_id) for job_id in raw_ids]
        has_more = len(job_ids) > limit
        items: list[dict[str, Any]] = []
        for job_id in job_ids[:limit]:
            data = await self._client.hgetall(keys.job_key(job_id))
            if data:
                items.append({str(k): str(v) for k, v in data.items()})
        return items, has_more

    async def is_doc_referenced(self, doc_id: str) -> bool:
        return bool(await self._client.exists(keys.doc_key(doc_id)))

    async def aclose(self) -> None:
        await self._client.aclose()
