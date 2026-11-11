"""Exception handlers mapping everything to the single error envelope."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.errors import ApiError, ErrorBody, ErrorCode, ErrorEnvelope

logger = structlog.get_logger("api.errors")

_STATUS_TO_CODE: dict[int, ErrorCode] = {
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    502: ErrorCode.MODEL_OUTPUT_INVALID,
    503: ErrorCode.DEPENDENCY_UNAVAILABLE,
    504: ErrorCode.DEADLINE_EXCEEDED,
}


def _envelope(
    request: Request, status_code: int, code: ErrorCode, message: str, retryable: bool
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorEnvelope(
        error=ErrorBody(code=code, message=message, retryable=retryable, request_id=request_id)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _envelope(request, exc.status_code, exc.code, exc.message, exc.retryable)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL)
        retryable = exc.status_code in (429, 503, 504)
        return _envelope(request, exc.status_code, code, str(exc.detail), retryable)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(part) for part in err.get('loc', []))}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        return _envelope(request, 422, ErrorCode.VALIDATION_ERROR, message, retryable=False)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_info=exc)
        return _envelope(
            request, 500, ErrorCode.INTERNAL, "internal server error", retryable=False
        )
