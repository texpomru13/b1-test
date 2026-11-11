"""External job/document/upload response contracts (IMPLEMENTATION_PLAN.md §9.1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.domain.job import JobStatus


class JobError(BaseModel):
    code: str
    message: str
    retryable: bool


class JobProgress(BaseModel):
    completed: int = 0
    total: int | None = None
    unit: str = "chunks"


class JobResponse(BaseModel):
    job_id: str
    doc_id: str
    status: JobStatus
    stage: str
    attempt: int
    progress: JobProgress
    created_at: str
    updated_at: str
    error: JobError | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> JobResponse:
        error: JobError | None = None
        if record.get("error_code"):
            error = JobError(
                code=record["error_code"],
                message=record.get("error_message", ""),
                retryable=record.get("error_retryable", "0") == "1",
            )
        return cls(
            job_id=record["job_id"],
            doc_id=record["doc_id"],
            status=JobStatus(record["status"]),
            stage=record.get("stage", "upload"),
            attempt=int(record.get("attempt", "0")),
            progress=JobProgress(
                completed=int(record.get("progress_completed", "0")),
                total=int(record["progress_total"]) if record.get("progress_total") else None,
                unit=record.get("progress_unit", "chunks"),
            ),
            created_at=record.get("created_at", ""),
            updated_at=record.get("updated_at", ""),
            error=error,
        )


class UploadAccepted(BaseModel):
    doc_id: str
    job_id: str
    status: JobStatus
    status_url: str
    document_url: str


class DocumentItem(BaseModel):
    doc_id: str
    title: str | None
    status: JobStatus
    ready: bool
    job_id: str
    created_at: str


class DocumentList(BaseModel):
    items: list[DocumentItem]
    next_cursor: str | None = None


class DocumentDetail(BaseModel):
    doc_id: str
    title: str | None
    original_filename: str | None
    content_type: str | None
    size_bytes: int | None
    status: JobStatus
    ready: bool
    job_id: str
    created_at: str
    updated_at: str
    page_count: int | None = None
    chunk_count: int | None = None
