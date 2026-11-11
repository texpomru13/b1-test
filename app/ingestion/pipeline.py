"""Parsing pipeline: stored file -> canonical text, page map, sections, chunks.

Produces the parse artifact `parse/parsed.json` (+ canonical.txt) inside the
document directory; stage 4 consumes it for embedding/indexing. All recorded
parameters are index-fingerprint inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from app.ingestion.canonical import NORMALIZATION_ID, CanonicalDocument, build_canonical
from app.ingestion.chunker import Chunk, LengthFn, split_into_chunks
from app.ingestion.docx_convert import convert_docx_to_pdf
from app.ingestion.pdf_loader import extract_pdf
from app.ingestion.sections import SectionSpan, detect_sections

PARSER_VERSION = "parse-v1"

CONTENT_TYPE_PDF = "application/pdf"
CONTENT_TYPE_DOCX = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class ParseResult:
    canonical: CanonicalDocument
    sections: list[SectionSpan]
    chunks: list[Chunk]
    page_count: int
    normalized_pdf: Path  # the PDF that page numbers refer to
    length_fn_id: str


def _noop_progress(_stage: str) -> None:
    return None


def parse_document(
    *,
    doc_dir: Path,
    content_type: str,
    max_pages: int,
    chunk_tokens: int,
    overlap_tokens: int,
    length_fn: LengthFn,
    length_fn_id: str,
    progress: ProgressFn = _noop_progress,
) -> ParseResult:
    if content_type == CONTENT_TYPE_PDF:
        pdf_path = doc_dir / "original.pdf"
    elif content_type == CONTENT_TYPE_DOCX:
        progress("convert")
        pdf_path = convert_docx_to_pdf(doc_dir / "original.docx", doc_dir / "normalized.pdf")
    else:  # unreachable behind FileStore validation; keep a clear error anyway
        from app.ingestion.errors import ParseFailedError

        raise ParseFailedError(f"unsupported content type '{content_type}'")

    progress("parse")
    extraction = extract_pdf(pdf_path, max_pages=max_pages)

    progress("normalize")
    canonical = build_canonical(extraction.pages)
    sections = detect_sections(canonical.text)

    progress("split")
    chunks = split_into_chunks(
        canonical,
        sections,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        length_fn=length_fn,
    )
    return ParseResult(
        canonical=canonical,
        sections=sections,
        chunks=chunks,
        page_count=extraction.page_count,
        normalized_pdf=pdf_path,
        length_fn_id=length_fn_id,
    )


def save_parse_artifacts(doc_dir: Path, result: ParseResult, *, chunk_tokens: int,
                         overlap_tokens: int) -> Path:
    parse_dir = doc_dir / "parse"
    parse_dir.mkdir(parents=True, exist_ok=True)
    (parse_dir / "canonical.txt").write_text(result.canonical.text, encoding="utf-8")
    payload = {
        "parser_version": PARSER_VERSION,
        "normalization": NORMALIZATION_ID,
        "length_fn": result.length_fn_id,
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "page_count": result.page_count,
        "normalized_pdf": result.normalized_pdf.name,
        "canonical_sha256": hashlib.sha256(
            result.canonical.text.encode("utf-8")
        ).hexdigest(),
        "pages": [asdict(span) for span in result.canonical.pages],
        "sections": [asdict(span) for span in result.sections],
        "chunks": [asdict(chunk) for chunk in result.chunks],
    }
    artifact = parse_dir / "parsed.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return artifact
