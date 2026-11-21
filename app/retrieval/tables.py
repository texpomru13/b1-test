"""Attach a verified price-column heading to contiguous numbered table rows."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.search import ContextChunk

_PRICE = re.compile(r"\b(?:цена|стоимость|тариф)\b", re.I)
_CURRENCY = re.compile(r"(?:\bруб(?:л(?:и|ей|ях))?|₽|\bтенге|\bевро)\s*[.\-–]?\s*"
                       r"(?:\(?\s*с\s+НДС\s*\)?)?\s*$", re.I)
_COLUMN = re.compile(r"(?:№|наименовани|номенклатур|ед\.\s*изм|единиц\w*\s+измерения)", re.I)
_ROW = re.compile(r"^\s*(\d{1,4}|[Зз])[.)]?\s+(?=[^\d])")
_AMOUNT = re.compile(r"\d(?:[\d \xa0\u202f]*\d)?(?:[.,]\d+)?\s*$")


def price_heading(text: str) -> tuple[int, int] | None:
    """Return only the printed column labels, excluding an adjoining footnote."""
    lines = text.splitlines(keepends=True)
    for width in (1, 2, 3):
        tail = "".join(lines[-width:]).rstrip()
        if _PRICE.search(tail) and _COLUMN.search(tail) and _CURRENCY.search(tail):
            start = sum(map(len, lines[:-width]))
            return start + len(tail) - len(tail.lstrip()), start + len(tail)
    return None


def _row_range(text: str) -> tuple[int, int] | None:
    lines = text.splitlines()
    if not lines or not _ROW.match(lines[0]):
        return None
    numbers: list[int] = []
    for line in lines:
        m = _ROW.match(line)
        if m and _AMOUNT.search(line[m.end():]):
            # An isolated Cyrillic З is a common OCR rendering of row number 3.
            n = 3 if m[1] in {"з", "З"} else int(m[1])
            if numbers and n != numbers[-1] + 1:
                return None
            numbers.append(n)
        elif not numbers:
            return None
    return (numbers[0], numbers[-1]) if numbers else None


def attach_price_headings(units: dict[str, list[ContextChunk]]) -> None:
    """Require real, consecutive source chunks and monotone row numbering."""
    heading = None
    bounds = None
    position = -2
    previous_row = 0
    for key, unit in sorted(units.items(), key=lambda item: item[1][-1].chunk_index):
        row = unit[-1]
        if row.chunk_index != position + 1 or (heading and row.scope_id != heading.scope_id):
            heading = None
        position = row.chunk_index
        label = price_heading(row.text)
        if label:
            heading = row
            bounds = row.char_start + label[0], row.char_start + label[1]
            previous_row = 0
            continue
        if heading is None:
            continue
        numbers = _row_range(row.text)
        if numbers is None or numbers[0] != previous_row + 1:
            heading = None
            continue
        previous_row = numbers[1]
        if len(unit) == 1:
            units[key] = [heading, replace(row, list_heading_id=heading.chunk_id,
                                           table_heading_span=bounds)]
