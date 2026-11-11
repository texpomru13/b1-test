"""Job lifecycle domain: states and stages per IMPLEMENTATION_PLAN.md §8.3 / ADR 003."""

from __future__ import annotations

import enum


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

# Allowed transitions (stage-2 subset is exercised; the full set arrives with stages 4/7).
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.ACCEPTED: frozenset({JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.VALIDATING, JobStatus.RETRY_WAIT, JobStatus.COMPENSATING,
         JobStatus.SUCCEEDED, JobStatus.FAILED}
    ),
    JobStatus.VALIDATING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.RETRY_WAIT, JobStatus.COMPENSATING, JobStatus.FAILED}
    ),
    JobStatus.RETRY_WAIT: frozenset({JobStatus.RUNNING, JobStatus.FAILED}),
    JobStatus.COMPENSATING: frozenset({JobStatus.FAILED, JobStatus.CLEANUP_PENDING}),
    JobStatus.CLEANUP_PENDING: frozenset({JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
}


class JobStage(enum.StrEnum):
    UPLOAD = "upload"
    STUB = "stub"  # stage-2 controlled stub; real stages (parse/split/embed/...) arrive later
    PARSE = "parse"
    NORMALIZE = "normalize"
    SPLIT = "split"
    EMBED = "embed"
    UPSERT = "upsert"
    VERIFY = "verify"
    PUBLISH = "publish"


# Job-level error codes (domain errors shown in GET /jobs/{id}).
ERROR_ATTEMPTS_EXHAUSTED = "ATTEMPTS_EXHAUSTED"
