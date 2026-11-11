"""Single error envelope contract: {"error": {code, message, retryable, request_id}}.

Source of truth for the external error format (IMPLEMENTATION_PLAN.md §9.3).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel


class ErrorCode(enum.StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    RATE_LIMITED = "RATE_LIMITED"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    INTERNAL = "INTERNAL"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    """Domain error carrying the HTTP status and envelope fields."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
