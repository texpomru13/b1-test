"""Select a matching tariff row with its own footnote or an explicit use-count rule."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.retrieval.tables import price_heading

if TYPE_CHECKING:
    from app.retrieval.search import ContextChunk

_PRICE_ASK = re.compile(r"\b(?:стоит|стоят|стоимост\w*|цен[ауые])\b", re.I)
_COMPARE = re.compile(r"\b(?:сравн\w*|разниц\w*|отлич\w*|соответственно|дешевле|дороже)\b", re.I)
_ROW = re.compile(r"^\s*\d{1,4}[.)]?\s+(.+?)\s+(\d+(?:[ \xa0\u202f]\d{3})*(?:[.,]\d+)?)\s*$")
_QUANTITY = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)[ \t]+([А-ЯЁа-яёA-Za-z]+)")
_MARK = re.compile(r"(?<!\*)\*{1,3}(?!\*)")
_NOTE = re.compile(r"(?m)^[ \t]*(\*{1,3})[ \t]+([^\n]+(?:\n[ \t]*[а-яё][^\n]*)*)")
_ASK_COUNT = re.compile(r"\b(?:сколько\s+раз|как\s+часто|кратност\w*)\b", re.I)
COUNT_RULE = re.compile(
    r"\b(?:\d+|один|два|три|четыре|единственный)\s+раз(?:а)?\b"
    r"|\bоднократно\b",
    re.I,
)


def _terms(text: str) -> set[str]:
    # The QA and retrieval packages share the same Russian word stemming.
    from app.qa.completeness import words

    return words(text) - {"скольк", "сто", "стоимост", "услуг", "быт", "како", "как", "цен"}


@dataclass(frozen=True)
class TariffRow:
    chunk: ContextChunk
    heading: ContextChunk
    label: str
    price: str
    note: tuple[ContextChunk, str] | None


def matched_tariff(question: str, chunks: list[ContextChunk]) -> TariffRow | None:
    if not _PRICE_ASK.search(question) or _COMPARE.search(question):
        return None
    query = _terms(question)
    quoted = re.findall(r'«([^»]+)»|"([^"]+)"', question)
    quantities = {(n, frozenset(_terms(u))) for n, u in _QUANTITY.findall(question)}
    by_id = {c.chunk_id: c for c in chunks}
    headings = sorted(c.chunk_index for c in chunks if price_heading(c.text))
    ranked = []
    for chunk in chunks:
        header = by_id.get(chunk.list_heading_id) if chunk.list_heading_id else None
        match = _ROW.fullmatch(chunk.text)
        if not header or not match or not chunk.table_heading_span:
            continue
        offset = chunk.table_heading_span[0] - header.char_start
        # Only the title immediately above these columns may identify a table.
        title = header.text[:offset].strip().splitlines()[-1:] if offset else []
        scope = len(query & _terms(" ".join(title)))
        label = match[1]
        overlap = len(query & _terms(label))
        next_header = next((index for index in headings if index > header.chunk_index), None)
        markers = set(_MARK.findall(label))
        notes = []
        if markers:
            for c in chunks:
                if not chunk.chunk_index <= c.chunk_index <= chunk.chunk_index + 48:
                    continue
                if next_header is not None and c.chunk_index > next_header:
                    continue
                if c.scope_id != chunk.scope_id:
                    continue
                # A marker appearing under a newer numbering sequence belongs
                # to the newer table, even if its column header was not retrieved.
                a = re.match(r"^\s*(\d+)[.)]?\s+", chunk.text)
                b = re.match(r"^\s*(\d+)[.)]?\s+", c.text)
                if not a or not b or int(b[1]) - int(a[1]) != c.chunk_index - chunk.chunk_index:
                    continue
                for m in _NOTE.finditer(c.text):
                    if m[1] in markers:
                        notes.append((c, m[2].strip()))
        note = notes[0] if len(notes) == 1 else None
        description = " ".join([*title, label, note[1] if note else ""])
        # An approximate product name cannot overrule a concrete size/quantity
        # or quoted product named in the question, even when no better row exists.
        if any(not _terms(left or right) <= _terms(description) for left, right in quoted):
            continue
        if quantities and not quantities <= {
            (n, frozenset(_terms(u))) for n, u in _QUANTITY.findall(description)
        }:
            continue
        note_score = len(query & _terms(note[1])) if note else 0
        if overlap < 3 and note_score < 3:
            continue
        # A footnote and its row carry distinct requested information. Nearby
        # rows without that marker receive no credit from the footnote.
        score = overlap + scope + (note_score if note_score >= 3 else 0)
        ranked.append(
            (score, TariffRow(chunk, header, label, match[2], note if note_score >= 3 else None))
        )
    if not ranked:
        return None
    score = max(value for value, _ in ranked)
    best = [row for value, row in ranked if value == score]
    return best[0] if len(best) == 1 else None


def is_count_question(question: str) -> bool:
    return bool(_ASK_COUNT.search(question))


def prioritize_source_details(
    question: str, ordered: list[Any], units: dict[str, list[ContextChunk]]
) -> list[Any]:
    if not ordered:
        return ordered
    chunks = {c.chunk_id: c for unit in units.values() for c in unit}
    keys = []
    match = matched_tariff(question, list(chunks.values()))
    if match:
        keys = [match.chunk.chunk_id]
        if match.note:
            keys.append(match.note[0].chunk_id)
    elif is_count_question(question):
        top = units[str(ordered[0].id)][-1]
        lead = re.match(r"\s*(\d+)\.", top.text)
        terms = _terms(top.text) & _terms(question)
        if lead:
            options = [
                p
                for p in ordered
                if (c := units[str(p.id)][-1]).scope_id == top.scope_id
                and re.match(r"\s*" + lead[1] + r"\.", c.text)
                and abs(c.chunk_index - top.chunk_index) <= 8
                and COUNT_RULE.search(c.text)
                and len(_terms(c.text) & terms) >= 2
            ]
            if len(options) == 1:
                keys = [str(ordered[0].id), str(options[0].id)]
    by_id = {str(p.id): p for p in ordered}
    if not keys:
        return ordered
    return [by_id[k] for k in dict.fromkeys(keys) if k in by_id] + [
        p for p in ordered if str(p.id) not in keys
    ]
