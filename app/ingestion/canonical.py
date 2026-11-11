"""Canonical document text with a page map.

Documented normalization (NORMALIZATION_ID), applied per page BEFORE the
canonical text is fixed, so every later offset refers to the normalized text:

1. Unicode NFC.
2. Line endings CRLF/CR -> LF.
3. Trailing whitespace stripped from every line.
4. Runs of 3+ newlines collapsed to exactly 2 (one blank line).
5. Every page contributes at least a trailing "\\n"; pages are concatenated
   directly, so page spans tile the canonical text: [start, end) half-open,
   zero-based Python unicode character offsets (never bytes).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.ingestion.pdf_loader import ExtractedPage

NORMALIZATION_ID = "nfc+lf+rstrip-lines+collapse-blank-v1"

_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_page_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text).strip("\n")
    return text + "\n"


@dataclass(frozen=True)
class PageSpan:
    page: int  # 1-based physical page number
    char_start: int
    char_end: int  # half-open
    has_text: bool


@dataclass(frozen=True)
class CanonicalDocument:
    text: str
    pages: list[PageSpan]

    def page_of(self, offset: int) -> int:
        """1-based page containing the given canonical offset."""
        for span in self.pages:
            if span.char_start <= offset < span.char_end:
                return span.page
        if self.pages and offset >= self.pages[-1].char_end:
            return self.pages[-1].page
        return 1

    def pages_overlapping(self, char_start: int, char_end: int) -> list[PageSpan]:
        return [
            span
            for span in self.pages
            if span.char_start < char_end and char_start < span.char_end
        ]


def build_canonical(pages: list[ExtractedPage]) -> CanonicalDocument:
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for page in pages:
        normalized = normalize_page_text(page.text)
        start = cursor
        cursor += len(normalized)
        parts.append(normalized)
        spans.append(
            PageSpan(page=page.page, char_start=start, char_end=cursor, has_text=page.has_text)
        )
    return CanonicalDocument(text="".join(parts), pages=spans)
