"""Deterministic provenance and consistency checks (§7.2 items 1-4).

1. Evidence chunk_id must be in the FINAL (post-budget) allowed context.
2. (Scope is guaranteed by retrieval: context comes only from the scoped query.)
3. The quote must match the chunk text exactly after ONE documented whitespace
   normalization (runs of whitespace -> single space); offsets and pages are
   recovered SERVER-side from the raw chunk span.
4. Numbers, percents, amounts and the working/calendar-days distinction in the
   claim text must be present in its evidence quotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk


@dataclass(frozen=True)
class LocatedQuote:
    chunk: ContextChunk
    quote_raw: str  # exact substring of the chunk text (and of canonical text)
    char_start: int  # canonical offsets
    char_end: int
    page: int
    page_to: int


class ProvenanceError(Exception):
    """A fabricated/foreign chunk, unlocatable quote, or inconsistent facts."""


def _ws_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(chunk: ContextChunk, quote: str) -> LocatedQuote:
    """Find the quote in the chunk under whitespace normalization; recover the
    RAW span and canonical offsets. Raises ProvenanceError if absent."""
    tokens = _ws_normalize(quote).split(" ")
    if not tokens or not tokens[0]:
        raise ProvenanceError("empty quote")
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, chunk.text)
    if match is None:
        raise ProvenanceError(
            f"quote is not a substring of chunk {chunk.chunk_id} "
            "(after whitespace normalization)"
        )
    start = chunk.char_start + match.start()
    end = chunk.char_start + match.end()
    page, page_to = _pages_for_span(chunk, start, end)
    return LocatedQuote(
        chunk=chunk, quote_raw=match.group(0),
        char_start=start, char_end=end, page=page, page_to=page_to,
    )


def _pages_for_span(chunk: ContextChunk, start: int, end: int) -> tuple[int, int]:
    if chunk.segments:
        page = page_to = chunk.page_from
        for segment in chunk.segments:
            if segment["char_start"] <= start < segment["char_end"]:
                page = int(segment["page"])
            if segment["char_start"] < end <= segment["char_end"]:
                page_to = int(segment["page"])
        return page, max(page, page_to)
    return chunk.page_from, chunk.page_to


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_SPACED_AMOUNT = re.compile(r"\d{1,3}(?:[   ]\d{3})+")


def _numeric_tokens(text: str) -> set[str]:
    cleaned = text
    tokens: set[str] = set()
    for match in _SPACED_AMOUNT.finditer(text):
        tokens.add(re.sub(r"[   ]", "", match.group(0)))
        cleaned = cleaned.replace(match.group(0), " ")
    for match in _NUMBER.finditer(cleaned):
        tokens.add(match.group(0).replace(",", "."))
    return tokens


def check_claim_consistency(claim: Claim, quotes: list[str]) -> None:
    """Rule checks: every number/percent/amount in the claim must occur in its
    evidence; working/calendar day kinds must not be swapped."""
    joined = " ".join(quotes)
    quote_numbers = _numeric_tokens(joined)
    missing = [
        token for token in _numeric_tokens(claim.text) if token not in quote_numbers
    ]
    if missing:
        raise ProvenanceError(
            f"claim states numbers {missing} that are absent from its evidence"
        )
    for kind in ("рабоч", "календарн"):
        if kind in claim.text.lower() and kind not in joined.lower():
            raise ProvenanceError(
                f"claim mentions '{kind}*' days but the evidence does not"
            )


def validate_claims(
    claims: list[Claim],
    context_by_id: dict[str, ContextChunk],
) -> dict[int, list[LocatedQuote]]:
    """Deterministic §7.2 checks for every claim; returns located quotes per claim.

    Raises ProvenanceError on ANY fabricated chunk id, unlocatable quote or
    numeric/day-kind inconsistency — the caller decides between refusal and the
    single allowed correction.
    """
    located: dict[int, list[LocatedQuote]] = {}
    for index, claim in enumerate(claims):
        quotes: list[LocatedQuote] = []
        for evidence in claim.evidence:
            # Canonicalize the server-issued id: prompts render it as [source-N],
            # so a bracketed echo is faithful, not fabricated.
            key = evidence.chunk_id.strip().strip("[]").strip()
            chunk = context_by_id.get(key)
            if chunk is None:
                raise ProvenanceError(
                    f"evidence references '{evidence.chunk_id}' which is not in the "
                    "allowed context (fabricated or truncated away)"
                )
            quotes.append(locate_quote(chunk, evidence.quote))
        check_claim_consistency(claim, [q.quote_raw for q in quotes])
        located[index] = quotes
    return located
