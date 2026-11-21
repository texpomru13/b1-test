"""Resolve directly requested details against complete, already-retrieved source passages."""

from __future__ import annotations

import re

from app.qa.completeness import words
from app.qa.grounding import LocatedQuote, locate_quote
from app.qa.numbers import ruble_heading
from app.qa.schemas import Claim, DraftAnswer
from app.retrieval.offer_details import COUNT_RULE, is_count_question, matched_tariff
from app.retrieval.search import ContextChunk

_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+")
_WEEKDAY = re.compile(
    r"\b(?:кажд\w*|по)\s+(?:понедельник|вторник|сред[ауые]|средам|четверг|"
    r"пятниц|суббот|воскресен)\w*",
    re.I,
)
_CLOCK = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3])[:.][0-5]\d(?!\d)")
_START = re.compile(r"\b(?:начал[оа]|начина\w*|открыва\w*)\b", re.I)
_ENTITLEMENT = re.compile(
    r"\b(?:дающ|предоставляющ)\w*\s+(?:"
    r"право|возможность)\b",
    re.I,
)
_PROCEDURE = re.compile(r"\bв\s+порядке,?\s+установленном\s+(?:разделом|пунктом|статьёй)", re.I)
_AFTER_TERM = re.compile(r"\bпо\s+истечении\s+(?:указанного|этого|данного)\s+срока", re.I)
_EXPIRY_ASK = re.compile(r"\b(?:когда|по\s+истечении|через\s+сколько)\b", re.I)
_WINDOW = re.compile(r"\bв\s+течение\s+\d", re.I)
_INTRO = re.compile(r"\b(?:следующ\w*\s+(?:случаях|условиях)|в\s+том\s+числе)\s*[:.]+$", re.I)
_SECOND_ASK = re.compile(
    r"\bи\s+(?:где|когда|как|сколько|на\s+сколько|через\s+сколько|"
    r"како\w*|что|кто|почему|можно|нужно|должен)\b",
    re.I,
)


def _plain(text: str) -> str:
    return " ".join(text.split())


def _citation(
    raw: str, key: str, spans: dict[str, LocatedQuote], wording: str | None = None
) -> Claim:
    # A separate physical line containing only a page number is not part of a provision.
    raw = re.sub(r"\n[ \t]*\d{1,4}\s*$", "", raw).strip()
    issued = [(label, span) for label, span in spans.items() if label.rsplit(".", 1)[0] == key]
    proof = locate_quote(issued[0][1].chunk, raw)
    existing = next(
        (k for k, q in issued if q.char_start == proof.char_start and q.char_end == proof.char_end),
        None,
    )
    if existing is None:
        numbers = [
            int(label.rsplit(".", 1)[1]) for label, _ in issued if label.rsplit(".", 1)[1].isdigit()
        ]
        existing = f"{key}.{max(numbers, default=0) + 1}"
        spans[existing] = proof
    sentence = _NUMBERED.sub("", _plain(raw), count=1)
    return Claim(text=(wording or sentence).rstrip(".; ") + ".", evidence=[existing])


def _cited_chunks(claim: Claim, spans: dict[str, LocatedQuote]) -> set[str]:
    return {
        q.chunk.chunk_id
        for k in claim.evidence
        if isinstance(k, str) and (q := spans.get(k.strip("[] ")))
    }


def _tariff(
    question: str, chunks: dict[str, tuple[str, ContextChunk]], spans: dict[str, LocatedQuote]
) -> list[Claim]:
    row = matched_tariff(question, [c for _, c in chunks.values()])
    if not row or not row.chunk.table_heading_span:
        return []
    left, right = row.chunk.table_heading_span
    header = row.heading.text[left - row.heading.char_start : right - row.heading.char_start]
    if not ruble_heading(header):
        return []
    # Only replace a compound draft when the additional question is a quantity
    # supplied by this row or by its own marked note.
    other = _SECOND_ASK.search(question)
    if other:
        suffix = question[other.start() :]
        if not re.match(r"и\s+(?:(?:на|через)\s+)?сколько\b", suffix, re.I):
            return []
        evidence = row.label + " " + (row.note[1] if row.note else "")
        if not (words(suffix) - {"скольк", "через", "рассчита"}) & words(evidence):
            return []
    label = re.sub(r"\s*\*+\s*", " ", row.label).strip()
    per_person = re.search(r"\s+чел\.?$", label, re.I)
    label = re.sub(r"\s+(?:шт|чел)\.?$", "", label, flags=re.I)
    statement = f"{label} — {row.price} руб." + (" за человека" if per_person else "")
    claims = [_citation(row.chunk.text, chunks[row.chunk.chunk_id][0], spans, statement)]
    if row.note:
        source, raw = row.note
        claims.append(_citation(raw, chunks[source.chunk_id][0], spans))
    return claims


def _schedule(
    question: str, claim: Claim, source: ContextChunk, key: str, spans: dict[str, LocatedQuote]
) -> Claim | None:
    if not _EXPIRY_ASK.search(question) or not _START.search(claim.text):
        return None
    clocks = set(_CLOCK.findall(claim.text))
    if len(clocks) != 1:
        return None
    for day in _WEEKDAY.finditer(source.text):
        tail = source.text[day.start() : day.start() + 430]
        time = _CLOCK.search(tail)
        if not time or time[0] not in clocks:
            continue
        end = time.end() + int(tail[time.end() : time.end() + 1] == ".")
        passage = tail[:end]
        if (
            len(words(question) & words(passage)) >= 3
            and _START.search(passage)
            and not re.search(r"\n\s*(?:\d+[.)]|[A-ZА-ЯЁ][^\n]{0,40}:)\s", passage)
        ):
            return _citation(passage, key, spans)
    return None


def _restore_clause(
    claim: Claim, source: ContextChunk, key: str, question: str, spans: dict[str, LocatedQuote]
) -> Claim | None:
    raw = source.text.strip()
    if len(raw) > 1300 or not _NUMBERED.match(raw):
        return None
    terms = words(claim.text)
    if len(words(raw) & terms) < max(5, len(terms) * 0.68):
        return None
    if _ENTITLEMENT.search(raw) and not _ENTITLEMENT.search(claim.text):
        return _citation(raw, key, spans)
    if (
        _EXPIRY_ASK.search(question)
        and _PROCEDURE.search(raw)
        and _AFTER_TERM.search(raw)
        and not _PROCEDURE.search(claim.text)
    ):
        return _citation(raw, key, spans)
    return None


def complete_source_details(
    question: str, draft: DraftAnswer, spans: dict[str, LocatedQuote]
) -> DraftAnswer:
    """All derived evidence is within the final context and goes through normal verification."""
    chunks = {q.chunk.chunk_id: (key.rsplit(".", 1)[0], q.chunk) for key, q in spans.items()}
    if tariff := _tariff(question, chunks, spans):
        return DraftAnswer(answerable=True, claims=tariff, missing_information=[])
    if not draft.answerable:
        return draft
    count_question = is_count_question(question)
    claims = []
    source_sections = set()
    expiry_sections = set()
    for original in draft.claims:
        if count_question and _INTRO.search(original.text):
            continue
        replacements = []
        for cid in _cited_chunks(original, spans):
            key, source = chunks[cid]
            ref = _NUMBERED.match(source.text)
            if ref:
                source_sections.add((source.scope_id, ref[1].split(".")[0]))
            replacement = _schedule(question, original, source, key, spans) or _restore_clause(
                original, source, key, question, spans
            )
            if replacement and replacement not in replacements:
                replacements.append(replacement)
                if ref and _AFTER_TERM.search(source.text):
                    expiry_sections.add((source.scope_id, ref[1]))
        claims.append(replacements[0] if len(replacements) == 1 else original)
    # Distinct frequency and maturity provisions often occupy the following numbered clause.
    extras = []
    for key, source in chunks.values():
        raw = source.text.strip()
        ref = _NUMBERED.match(raw)
        if not ref or len(raw) > 1400 or len(words(raw) & words(question)) < 3:
            continue
        frequency = (
            count_question
            and (source.scope_id, ref[1].split(".")[0]) in source_sections
            and COUNT_RULE.search(raw)
            and not any(COUNT_RULE.search(c.text) for c in claims)
        )
        maturity = (
            source.scope_id,
            ref[1].rpartition(".")[0],
        ) in expiry_sections and _WINDOW.search(raw)
        if frequency or maturity:
            extras.append(_citation(raw, key, spans))
    if len(extras) == 1:
        claims.extend(extras)
    if not claims:
        return draft
    result = []
    for claim in claims:
        if claim not in result:
            result.append(claim)
    return DraftAnswer(answerable=True, claims=result, missing_information=[])
