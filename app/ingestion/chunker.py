"""LangChain section-bounded chunks with exact character offsets in the source."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.canonical import CanonicalDocument
from app.ingestion.errors import ParseFailedError
from app.ingestion.rules import conditional_rules, parenthesized_items
from app.ingestion.sections import SectionSpan

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
    ocr: bool = False


@dataclass(frozen=True)
class GoverningSpan:
    chunk_index: int
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
    scope_id: str | None = None
    scope_label: str | None = None
    scope_label_span: tuple[int, int] | None = None
    parent_clause_id: int | None = None
    governing_spans: tuple[GoverningSpan, ...] = field(default_factory=tuple)
    governing_complete: bool = True


class _SourcePiece(str):
    start: int


class _OffsetSplitter(RecursiveCharacterTextSplitter):
    """Track the actual pieces LangChain reuses for overlap, even with identical text."""

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        pieces: list[str] = []
        offset = 0
        for text in splits:
            piece = _SourcePiece(text)
            piece.start = offset
            pieces.append(piece)
            offset += len(piece)
        return super()._merge_splits(pieces, separator)

    def _join_docs(self, docs: list[str], separator: str) -> str:
        # keep_separator=True includes separators in the pieces themselves.
        first = docs[0]
        assert isinstance(first, _SourcePiece)
        chunk = _SourcePiece(separator.join(docs))
        chunk.start = first.start
        return chunk

    def spans(self, source: str) -> Iterator[tuple[int, str]]:
        end = base = 0
        for raw in self.split_text(source):
            # Each recursively merged group starts at zero. An unmerged leaf
            # (a single character over the token limit) is itself a new group.
            local = raw.start if isinstance(raw, _SourcePiece) else 0
            if local == 0:
                base = end
            start = base + local
            end = start + len(raw)
            if source[start:end] != raw:
                raise ParseFailedError("chunk offsets disagree with the canonical text")
            trimmed = raw.strip()
            if trimmed:
                yield start + len(raw) - len(raw.lstrip()), trimmed


def _split_rule_segment(
    splitter: _OffsetSplitter, text: str, length: LengthFn, limit: int,
) -> Iterator[tuple[int, str, bool]]:
    if length(text) <= limit:
        for local, chunk in splitter.spans(text):
            yield local, chunk, True
        return
    pos = 0
    for start, end in conditional_rules(text):
        for local, chunk in splitter.spans(text[pos:start]):
            yield pos + local, chunk, True
        rule = text[start:end]
        if length(rule) <= limit:
            yield start, rule, True
        else:
            for local, chunk in splitter.spans(rule):
                yield start + local, chunk, False
        pos = end
    for local, chunk in splitter.spans(text[pos:]):
        yield pos + local, chunk, True


def _split_rules(
    splitter: _OffsetSplitter, text: str, length: LengthFn, limit: int,
) -> Iterator[tuple[int, str, bool]]:
    boundaries = sorted({0, len(text), *parenthesized_items(text)})
    for begin, end in pairwise(boundaries):
        for start, chunk, complete in _split_rule_segment(splitter, text[begin:end], length, limit):
            yield begin + start, chunk, complete


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
    splitter = _OffsetSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=length_fn,
        strip_whitespace=False,
    )
    chunks: list[Chunk] = []
    by_start = {s.char_start: s for s in sections}
    leads: dict[int, GoverningSpan] = {}
    for seg_start, seg_end in _segment_bounds(canonical, sections):
        section = by_start.get(seg_start)
        scope = section.scope if section else None
        ancestors = section.governing_clause_ids if section else ()
        governing = tuple(leads[pos] for pos in ancestors if pos in leads)
        first = len(chunks)
        segment_text = canonical.text[seg_start:seg_end]
        if not segment_text.strip():
            continue
        for local_start, chunk_text, complete in _split_rules(
            splitter, segment_text, length_fn, chunk_tokens,
        ):
            start = seg_start + local_start
            end = start + len(chunk_text)
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
                    section=section.label if section else None,
                    scope_id=scope.scope_id if scope else None,
                    scope_label=scope.scope_label if scope else None,
                    scope_label_span=scope.scope_label_span if scope else None,
                    parent_clause_id=section.parent_clause_id if section else None,
                    governing_spans=governing,
                    governing_complete=complete and len(governing) == len(ancestors),
                    segments=[
                        ChunkSegment(
                            page=span.page, ocr=span.ocr,
                            char_start=max(start, span.char_start),
                            char_end=min(end, span.char_end),
                        )
                        for span in pages
                    ],
                )
            )
        # Store a complete introductory clause as a citable component. Never
        # truncate a long/split condition and then present it as authoritative.
        if len(chunks) == first + 1 and len(segment_text.strip()) <= 600:
            chunk = chunks[first]
            leads[seg_start] = GoverningSpan(first, chunk.char_start, chunk.char_end)
    return chunks
