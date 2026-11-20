"""Document titles and public indexing states."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from pathlib import PureWindowsPath
from typing import Any


class JobStatus(enum.StrEnum):
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    RETRY_WAIT = "retry_wait"
    COMPENSATING = "compensating"
    FAILED = "failed"
    CLEANUP_PENDING = "cleanup_pending"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED})

def document_title(record: Mapping[str, Any]) -> str:
    if title := record.get("title"):
        return str(title)
    filename = PureWindowsPath(record.get("original_filename") or "").name.strip()
    return filename or str(record["doc_id"])


class JobStage(enum.StrEnum):
    UPLOAD = "upload"
    PARSE = "parse"
    NORMALIZE = "normalize"
    SPLIT = "split"
    EMBED = "embed"
    UPSERT = "upsert"
    VERIFY = "verify"
    PUBLISH = "publish"


# Job-level error codes (domain errors shown in GET /jobs/{id}).
ERROR_ATTEMPTS_EXHAUSTED = "ATTEMPTS_EXHAUSTED"
