"""Extract physical pages and their text layer using pypdf.

Physical pages are numbered from one. The text layer determines whether a
document can be indexed directly or requires OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.ingestion.errors import (
    OcrRequiredError,
    PageLimitExceededError,
    ParseFailedError,
    PasswordProtectedError,
)

# Ignore short decorations on an image-only page.
MIN_TEXT_CHARS_PER_PAGE = 20
# Reject a mixed document when at least two pages and over 20% lack a text layer.
MIXED_EMPTY_RATIO = 0.2


@dataclass(frozen=True)
class ExtractedPage:
    page: int  # 1-based physical page of THIS pdf
    text: str
    has_text: bool


@dataclass(frozen=True)
class PdfExtraction:
    pages: list[ExtractedPage]
    page_count: int


def _open_reader(path: Path) -> PdfReader:
    try:
        reader = PdfReader(str(path))
    except PyPdfError as exc:
        raise ParseFailedError(f"cannot parse PDF: {type(exc).__name__}") from exc
    except OSError as exc:
        raise ParseFailedError(f"cannot read PDF file: {type(exc).__name__}") from exc
    if reader.is_encrypted:
        raise PasswordProtectedError("PDF is password-protected; upload a decrypted file")
    return reader


def extract_pdf(path: Path, *, max_pages: int) -> PdfExtraction:
    reader = _open_reader(path)
    try:
        page_count = len(reader.pages)
    except PyPdfError as exc:
        raise ParseFailedError(f"cannot enumerate PDF pages: {type(exc).__name__}") from exc
    if page_count > max_pages:
        raise PageLimitExceededError(
            f"document has {page_count} pages; the limit is {max_pages}"
        )

    pages: list[ExtractedPage] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (PyPdfError, ValueError, KeyError) as exc:
            raise ParseFailedError(
                f"text extraction failed on page {number}: {type(exc).__name__}"
            ) from exc
        meaningful = len("".join(text.split()))
        # A short text-only page may consist of a date, name or amount. For an
        # image page, a footer/page number alone is not a usable text layer.
        has_text = meaningful >= MIN_TEXT_CHARS_PER_PAGE or (meaningful > 0 and not page.images)
        pages.append(ExtractedPage(page=number, text=text, has_text=has_text))

    _check_text_layer(pages)
    return PdfExtraction(pages=pages, page_count=page_count)


def _check_text_layer(pages: list[ExtractedPage]) -> None:
    total = len(pages)
    empty = sum(1 for page in pages if not page.has_text)
    if total == 0 or empty == total:
        raise OcrRequiredError(
            "document has no usable text layer (scan?); OCR is not part of the "
            "mandatory scope",
            detail=f"pages_with_text=0/{total}",
        )
    if empty >= 2 and empty / total > MIXED_EMPTY_RATIO:
        raise OcrRequiredError(
            "document mixes text pages with scan-only pages; indexing only the "
            "text part would silently misrepresent the document",
            detail=f"pages_with_text={total - empty}/{total}",
        )
