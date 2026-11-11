"""Server-side answer assembly (§7.5): short claims with [n] markers; citations
derived from located quotes (authoritative chunk payload + recovered offsets),
deduplicated by span with stable ordering.
"""

from __future__ import annotations

from app.qa.graph import QaState
from app.qa.grounding import LocatedQuote
from app.schemas.qa import AnswerResponse, Citation


def assemble_answer(state: QaState) -> AnswerResponse:
    draft = state["draft"]
    located = state["located"]
    assert draft is not None and located is not None

    citations: list[Citation] = []
    by_span: dict[tuple[str, int, int], int] = {}

    def citation_id(quote: LocatedQuote) -> int:
        key = (quote.chunk.chunk_id, quote.char_start, quote.char_end)
        if key in by_span:
            return by_span[key]
        next_id = len(citations) + 1
        citations.append(
            Citation(
                id=next_id,
                text=quote.quote_raw,
                page=quote.page,
                page_to=quote.page_to,
                section=quote.chunk.section,
                chunk_id=quote.chunk.chunk_id,
                char_start=quote.char_start,
                char_end=quote.char_end,
            )
        )
        by_span[key] = next_id
        return next_id

    parts: list[str] = []
    for index, claim in enumerate(draft.claims):
        markers = "".join(f"[{citation_id(q)}]" for q in located[index])
        text = claim.text.strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        parts.append(f"{text} {markers}".strip())

    return AnswerResponse(answer=" ".join(parts), found=True, citations=citations)
