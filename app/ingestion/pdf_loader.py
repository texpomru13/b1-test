"""PDF text extraction: a custom LangChain BaseLoader over pypdf.

Physical, 1-based pages. Encrypted/corrupt inputs raise typed errors; the
text-layer statistics feed the mandatory scan/mixed detection (stage 3).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.ingestion.errors import (
    OcrRequiredError,
    PageLimitExceededError,
    ParseFailedError,
    PasswordProtectedError,
)

# A page with fewer meaningful characters than this has no usable text layer.
MIN_TEXT_CHARS_PER_PAGE = 20
# Mixed-document rule: flag when >20% of pages (or >=2 pages in small docs) lack text.
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
        pages.append(
            ExtractedPage(page=number, text=text, has_text=meaningful >= MIN_TEXT_CHARS_PER_PAGE)
        )

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


class PdfPagesLoader(BaseLoader):
    """LangChain loader: one Document per physical page (metadata: page, 1-based)."""

    def __init__(self, path: Path, *, max_pages: int) -> None:
        self._path = path
        self._max_pages = max_pages

    def lazy_load(self) -> Iterator[Document]:
        extraction = extract_pdf(self._path, max_pages=self._max_pages)
        for page in extraction.pages:
            yield Document(
                page_content=page.text,
                metadata={"page": page.page, "source": str(self._path)},
            )
