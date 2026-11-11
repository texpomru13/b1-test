"""Tokenizer-aware chunking with exact canonical offsets.

LangChain RecursiveCharacterTextSplitter runs INSIDE section-bounded segments
(plan §5.2). Offsets are computed by OUR monotonic search within the parent
segment: find(chunk, previous_chunk_start + 1) — never a global text.find, and
never LangChain's add_start_index (its offset estimate mixes token-based overlap
with character arithmetic and can miss tail chunks in repeated text). Repeated
paragraphs and identical clauses therefore map to strictly increasing offsets.
Invariant (verified for every chunk): canonical[char_start:char_end] == chunk.text.

Length functions:
- HFTokenizerLength — exact, via the `tokenizers` package (no torch) from a
  tokenizer.json provisioned with the resolved embedding model.
- ApproxTokenLength — documented conservative fallback for environments where
  the tokenizer asset is not yet provisioned: ceil(chars / 3) (Cyrillic BPE runs
  ~3 chars/token); the exact budget is re-enforced at embed time in stage 4.
The chosen function id is recorded in the parse artifact (index fingerprint input).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.canonical import CanonicalDocument
from app.ingestion.errors import ParseFailedError
from app.ingestion.sections import SectionSpan, section_at

APPROX_CHARS_PER_TOKEN = 3


class ApproxTokenLength:
    id = f"approx-v1:cpt{APPROX_CHARS_PER_TOKEN}"

    def __call__(self, text: str) -> int:
        return math.ceil(len(text) / APPROX_CHARS_PER_TOKEN)


class HFTokenizerLength:
    """Exact token count using a HuggingFace tokenizer.json (tokenizers package)."""

    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer  # local import: optional asset

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.id = f"hf:{tokenizer_path.name}"

    def __call__(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


LengthFn = Callable[[str], int]


@dataclass(frozen=True)
class ChunkSegment:
    page: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int
    page_from: int
    page_to: int
    section: str | None
    segments: list[ChunkSegment]


def _segment_bounds(
    canonical: CanonicalDocument, sections: list[SectionSpan]
) -> list[tuple[int, int]]:
    """Section-bounded segments tiling the canonical text (preamble included)."""
    if not sections:
        return [(0, len(canonical.text))] if canonical.text else []
    bounds: list[tuple[int, int]] = []
    if sections[0].char_start > 0:
        bounds.append((0, sections[0].char_start))
    bounds.extend((span.char_start, span.char_end) for span in sections)
    return bounds


def split_into_chunks(
    canonical: CanonicalDocument,
    sections: list[SectionSpan],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
    length_fn: LengthFn,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=length_fn,
        strip_whitespace=True,
    )
    chunks: list[Chunk] = []
    for seg_start, seg_end in _segment_bounds(canonical, sections):
        segment_text = canonical.text[seg_start:seg_end]
        if not segment_text.strip():
            continue
        previous_local = -1
        for chunk_text in splitter.split_text(segment_text):
            local_start = segment_text.find(chunk_text, previous_local + 1)
            if local_start < 0:
                raise ParseFailedError("splitter chunk not found in its parent segment")
            previous_local = local_start
            start = seg_start + local_start
            end = start + len(chunk_text)
            if canonical.text[start:end] != chunk_text:
                raise ParseFailedError(
                    f"chunk offset invariant violated at canonical[{start}:{end}]"
                )
            pages = canonical.pages_overlapping(start, end)
            if not pages:
                raise ParseFailedError(f"chunk [{start}:{end}] maps to no page span")
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    page_from=pages[0].page,
                    page_to=pages[-1].page,
                    section=section_at(sections, start),
                    segments=[
                        ChunkSegment(
                            page=span.page,
                            char_start=max(start, span.char_start),
                            char_end=min(end, span.char_end),
                        )
                        for span in pages
                    ],
                )
            )
    return chunks
