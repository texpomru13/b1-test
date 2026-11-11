"""Upload + document routes (stage 2: lifecycle registry only; Qdrant merge at stage 4)."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from app.domain.fingerprint import upload_fingerprint
from app.domain.job import TERMINAL_STATUSES, JobStatus
from app.schemas.errors import ApiError, ErrorCode
from app.schemas.jobs import (
    DocumentDetail,
    DocumentItem,
    DocumentList,
    UploadAccepted,
)
from app.storage.file_store import (
    FileStore,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])
logger = structlog.get_logger("api.documents")

MAX_TITLE_CHARS = 500


def _repo(request: Request) -> Any:
    return request.app.state.redis_repo


def _file_store(request: Request) -> FileStore:
    store = getattr(request.app.state, "file_store", None)
    if store is None:
        raise ApiError(503, ErrorCode.DEPENDENCY_UNAVAILABLE, "file store is not ready",
                       retryable=True)
    return store


@router.post("/documents", status_code=202, response_model=UploadAccepted)
async def upload_document(
    request: Request,
    file: UploadFile,
    title: Annotated[str | None, Form()] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    settings = request.app.state.settings
    store = _file_store(request)

    clean_title = (title or "").strip() or None
    if clean_title and len(clean_title) > MAX_TITLE_CHARS:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR,
                       f"title exceeds {MAX_TITLE_CHARS} characters")

    doc_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex

    try:
        stored = await store.save(
            doc_id=doc_id,
            content_type=file.content_type or "",
            read=file.read,
        )
    except FileTooLargeError as exc:
        raise ApiError(413, ErrorCode.PAYLOAD_TOO_LARGE, str(exc)) from exc
    except UnsupportedFileTypeError as exc:
        raise ApiError(415, ErrorCode.UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

    fingerprint = upload_fingerprint(
        file_sha256=stored.sha256, title=clean_title, settings=settings
    )
    idem = idempotency_key or f"auto-{fingerprint}"

    try:
        result = await _repo(request).submit(
            idem_key=idem,
            fingerprint=fingerprint,
            job_id=job_id,
            doc_id=doc_id,
            job_fields={
                "title": clean_title or "",
                "original_filename": file.filename or "",
                "content_type": stored.content_type,
                "file_sha256": stored.sha256,
                "size_bytes": str(stored.size_bytes),
                "fingerprint": fingerprint,
                "idem_key": idem,
            },
        )
    except Exception as exc:
        # File stays on disk; the janitor removes unreferenced docs after the grace
        # period (plan §8.2: file+Redis cannot commit atomically -> 503, retry safe).
        logger.warning("submit_failed_redis", doc_id=doc_id, error=type(exc).__name__)
        raise ApiError(
            503, ErrorCode.DEPENDENCY_UNAVAILABLE,
            "job registry is unavailable; retry the upload with the same Idempotency-Key",
            retryable=True,
        ) from exc

    if result.verdict == "CONFLICT":
        store.delete_doc_files(doc_id)
        raise ApiError(
            409, ErrorCode.CONFLICT,
            "Idempotency-Key was already used with a different file/title/settings",
        )
    if result.verdict == "DUPLICATE":
        store.delete_doc_files(doc_id)  # keep the original upload's files only

    status_url = f"/api/v1/jobs/{result.job_id}"
    document_url = f"/api/v1/documents/{result.doc_id}"
    body = UploadAccepted(
        doc_id=result.doc_id,
        job_id=result.job_id,
        status=JobStatus.ACCEPTED,
        status_url=status_url,
        document_url=document_url,
    )
    return JSONResponse(
        status_code=202,
        content=body.model_dump(mode="json"),
        headers={"Location": status_url},
    )


def _doc_item(record: dict[str, Any]) -> DocumentItem:
    status = JobStatus(record["status"])
    return DocumentItem(
        doc_id=record["doc_id"],
        title=record.get("title") or None,
        status=status,
        # Stage 2: nothing is indexed yet; ready requires succeeded + published run
        # (completed at stage 4 with the Qdrant head-point merge).
        ready=status == JobStatus.SUCCEEDED,
        job_id=record["job_id"],
        created_at=record.get("created_at", ""),
    )


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    request: Request, limit: int = 20, cursor: str | None = None
) -> DocumentList:
    if not 1 <= limit <= 100:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "limit must be between 1 and 100")
    try:
        offset = int(cursor) if cursor else 0
        if offset < 0:
            raise ValueError
    except ValueError as exc:
        raise ApiError(422, ErrorCode.VALIDATION_ERROR, "invalid cursor") from exc

    records, has_more = await _repo(request).list_jobs(offset=offset, limit=limit)
    items = [_doc_item(record) for record in records]
    return DocumentList(items=items, next_cursor=str(offset + limit) if has_more else None)


@router.get("/documents/{doc_id}/normalized")
async def get_normalized_pdf(request: Request, doc_id: str) -> Any:
    """The server-side PDF that citation page numbers refer to (plan §5.1)."""
    from fastapi.responses import FileResponse

    job_id = await _repo(request).get_job_id_for_doc(doc_id)
    if not job_id:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"document '{doc_id}' is unknown")
    store = _file_store(request)
    doc_dir = store.doc_dir(doc_id)
    for name in ("normalized.pdf", "original.pdf"):
        candidate = doc_dir / name
        if candidate.exists():
            return FileResponse(candidate, media_type="application/pdf",
                                filename=f"{doc_id}.pdf")
    raise ApiError(
        404, ErrorCode.NOT_FOUND,
        "normalized PDF is not available for this document (conversion not finished "
        "or the job failed before parsing)",
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document(request: Request, doc_id: str) -> DocumentDetail:
    job_id = await _repo(request).get_job_id_for_doc(doc_id)
    if not job_id:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"document '{doc_id}' is unknown")
    record = await _repo(request).get_job(job_id)
    if not record:
        raise ApiError(404, ErrorCode.NOT_FOUND, f"document '{doc_id}' is unknown")
    status = JobStatus(record["status"])
    return DocumentDetail(
        doc_id=doc_id,
        title=record.get("title") or None,
        original_filename=record.get("original_filename") or None,
        content_type=record.get("content_type") or None,
        size_bytes=int(record["size_bytes"]) if record.get("size_bytes") else None,
        status=status,
        ready=status == JobStatus.SUCCEEDED and status in TERMINAL_STATUSES,
        job_id=job_id,
        created_at=record.get("created_at", ""),
        updated_at=record.get("updated_at", ""),
        page_count=int(record["page_count"]) if record.get("page_count") else None,
        chunk_count=int(record["chunk_count"]) if record.get("chunk_count") else None,
    )
