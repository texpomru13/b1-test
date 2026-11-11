"""Typed parsing errors mapped to job error codes (plan §8.4 / §9.3)."""

from __future__ import annotations


class ParseError(Exception):
    """Terminal or retryable document-processing failure with a stable code."""

    code: str = "PARSE_FAILED"
    retryable: bool = False

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class PageLimitExceededError(ParseError):
    code = "PAGE_LIMIT_EXCEEDED"


class PasswordProtectedError(ParseError):
    code = "PASSWORD_PROTECTED"


class OcrRequiredError(ParseError):
    """No or insufficient text layer (including mixed scan/text documents)."""

    code = "OCR_REQUIRED"


class ParseFailedError(ParseError):
    code = "PARSE_FAILED"


class ConversionFailedError(ParseError):
    code = "CONVERSION_FAILED"


class ConversionTimeoutError(ParseError):
    code = "CONVERSION_TIMEOUT"
    retryable = True
