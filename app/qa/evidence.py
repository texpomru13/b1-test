"""Issue request-local evidence IDs for contiguous spans of the allowed chunks."""

from __future__ import annotations

import re
from itertools import pairwise

from app.ingestion.rules import parenthesized_items
from app.qa.grounding import LocatedQuote, quote_at_offsets
from app.qa.numbers import DOCUMENT_TITLE
from app.retrieval.search import ContextChunk

# A physical PDF line is often only half a sentence. Split before numbered or
# lettered clauses, between paragraphs, or after a sentence ending. A period
# terminating a clause number or short abbreviation is not a sentence ending.
_BOUNDARY = re.compile(
    r"\n[ \t]*\n\s*|\n\s*(?=(?:\d+(?:\.\d+)+[.)]?|[а-яa-hj-uwyz][)])\s)"
    r"|(?<=[!?])\s+(?=[А-ЯЁA-Z])|(?<=[а-яёa-z]{3}\.)\s+(?=[А-ЯЁA-Z])"
)
_CONTINUATION = re.compile(
    r"(?:за исключением|исключени\w*|кроме|однако|при этом|в последнем случае)\b", re.I
)
_MAX_SPANS_PER_CHUNK = 12


def issue_evidence(sources: dict[str, ContextChunk]) -> dict[str, LocatedQuote]:
    spans: dict[str, LocatedQuote] = {}
    for source_id, chunk in sources.items():
        boundaries = [0]
        for match in _BOUNDARY.finditer(chunk.text):
            if not _CONTINUATION.match(chunk.text, match.end()):
                boundaries.append(match.end())
        # Also support source chunks indexed before parenthesized articles
        # became separate retrieval units. These spans remain inside the budget.
        boundaries = sorted({*boundaries, *parenthesized_items(chunk.text)})
        # Bound the label overhead for tables and very short paragraphs.
        stride = (len(boundaries) + _MAX_SPANS_PER_CHUNK - 1) // _MAX_SPANS_PER_CHUNK
        # A compilation can contain the end of one instrument and the title
        # of the next in a single chunk. Give the actual title its own ID.
        titles = [pos for title in DOCUMENT_TITLE.finditer(chunk.text) for pos in title.span()]
        boundaries = sorted({*boundaries[::stride], *titles, len(chunk.text)})
        for number, (start, end) in enumerate(pairwise(boundaries), 1):
            raw = chunk.text[start:end]
            if raw.strip():
                spans[f"{source_id}.{number}"] = quote_at_offsets(
                    chunk, start + len(raw) - len(raw.lstrip()), start + len(raw.rstrip())
                )
    return spans
