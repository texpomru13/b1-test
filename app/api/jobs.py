"""Job status route: reads OUR lifecycle registry, never Celery AsyncResult."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.schemas.errors import ApiError, ErrorCode
from app.schemas.jobs import JobResponse

router = APIRouter(prefix="/api/v1", tags=["jobs"])


def _repo(request: Request) -> Any:
    return request.app.state.redis_repo


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(request: Request, job_id: str) -> JobResponse:
    record = await _repo(request).get_job(job_id)
    if record is None:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"job '{job_id}' is unknown")
    return JobResponse.from_record(record)
