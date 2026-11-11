from pathlib import Path

import pytest
from app.ingestion.canonical import build_canonical, normalize_page_text
from app.ingestion.chunker import ApproxTokenLength, split_into_chunks
from app.ingestion.errors import (
    OcrRequiredError,
    PageLimitExceededError,
    ParseFailedError,
    PasswordProtectedError,
)
from app.ingestion.pdf_loader import ExtractedPage, extract_pdf
from app.ingestion.pipeline import parse_document, save_parse_artifacts
from app.ingestion.sections import detect_sections, section_at

from tests.helpers.pdfgen import (
    make_corrupt_pdf,
    make_encrypted_pdf,
    make_text_pdf,
)

# --- canonical ---


def test_normalization_rules() -> None:
    raw = "Строка  \r\nвторая\rтретья\n\n\n\nконец"
    assert normalize_page_text(raw) == "Строка\nвторая\nтретья\n\nконец\n"


def test_page_map_tiles_canonical() -> None:
    pages = [
        ExtractedPage(page=1, text="Первая страница текста договора.", has_text=True),
        ExtractedPage(page=2, text="Вторая страница текста договора.", has_text=True),
    ]
    doc = build_canonical(pages)
    assert doc.pages[0].char_start == 0
    assert doc.pages[0].char_end == doc.pages[1].char_start
    assert doc.pages[-1].char_end == len(doc.text)
    assert doc.page_of(0) == 1
    assert doc.page_of(doc.pages[1].char_start) == 2


# --- sections ---

CONTRACT_TEXT = """Договор поставки № АБ-123/24

1. Предмет договора
Поставщик обязуется поставить товар.

4.2 Штрафные санкции
Неустойка составляет 0,1% в день.
Сумма 1 000 000 (один миллион) рублей.
4.20 рублей не является номером раздела в середине строки.

4.2.1 Ограничение ответственности
Общая сумма неустойки не превышает 10% цены.

Приложение № 1
Спецификация товара.
"""


def test_section_detection_russian() -> None:
    sections = detect_sections(CONTRACT_TEXT)
    labels = [s.label for s in sections]
    assert "1. Предмет договора" in labels
    assert "4.2 Штрафные санкции" in labels
    assert "4.2.1 Ограничение ответственности" in labels
    assert "Приложение № 1" in labels
    # decimals/amounts are not sections
    assert not any("0,1%" in label for label in labels)
    assert not any(label.startswith("Сумма") for label in labels)
    assert not any(label.startswith("4.20") for label in labels)  # lowercase after number


def test_preamble_has_no_invented_section() -> None:
    sections = detect_sections(CONTRACT_TEXT)
    assert section_at(sections, 0) is None  # title line before the first heading


# --- chunker: offsets, repeated text, multi-page ---


def test_chunk_offsets_with_repeated_paragraphs() -> None:
    clause = "Стороны несут ответственность в соответствии с законодательством.\n"
    text1 = "1. Раздел первый\n" + clause * 3
    text2 = "2. Раздел второй\n" + clause * 3
    pages = [
        ExtractedPage(page=1, text=text1, has_text=True),
        ExtractedPage(page=2, text=text2, has_text=True),
    ]
    doc = build_canonical(pages)
    sections = detect_sections(doc.text)
    chunks = split_into_chunks(
        doc, sections, chunk_tokens=40, overlap_tokens=5, length_fn=ApproxTokenLength()
    )
    assert len(chunks) >= 2
    previous_start = -1
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text  # invariant
        assert chunk.char_start > previous_start or chunk.char_start >= 0
        previous_start = chunk.char_start
    # identical clause appears in both sections -> same text, different offsets
    texts = [c.text for c in chunks]
    assert len(set(c.char_start for c in chunks)) == len(chunks)
    assert any(chunks[i].section != chunks[j].section
               for i in range(len(chunks)) for j in range(i + 1, len(chunks))
               if texts[i] == texts[j]) or len(set(texts)) == len(texts)


def test_multi_page_chunk_has_segments() -> None:
    long_text = "Пункт договора о поставке товара надлежащего качества. " * 10
    pages = [
        ExtractedPage(page=1, text=long_text, has_text=True),
        ExtractedPage(page=2, text=long_text, has_text=True),
    ]
    doc = build_canonical(pages)
    chunks = split_into_chunks(
        doc, [], chunk_tokens=10_000, overlap_tokens=0, length_fn=ApproxTokenLength()
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert (chunk.page_from, chunk.page_to) == (1, 2)
    assert [seg.page for seg in chunk.segments] == [1, 2]
    assert chunk.segments[0].char_end == doc.pages[0].char_end
    assert chunk.segments[1].char_start == doc.pages[1].char_start


# --- pdf extraction on generated fixtures ---


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_pdf_text_extraction(tmp_path: Path) -> None:
    pdf = _write(tmp_path, "doc.pdf", make_text_pdf(
        ["Section 4.2 Penalty terms apply here on page one.",
         "Second page continues the contract text."]
    ))
    extraction = extract_pdf(pdf, max_pages=100)
    assert extraction.page_count == 2
    assert "Penalty terms" in extraction.pages[0].text
    assert extraction.pages[0].page == 1 and extraction.pages[1].page == 2


def test_page_limit_101_rejected(tmp_path: Path) -> None:
    pages = [f"Page {i} of a long synthetic contract document." for i in range(1, 102)]
    pdf = _write(tmp_path, "big.pdf", make_text_pdf(pages))
    with pytest.raises(PageLimitExceededError):
        extract_pdf(pdf, max_pages=100)
    # 100 pages pass
    pdf_ok = _write(tmp_path, "ok.pdf", make_text_pdf(pages[:100]))
    assert extract_pdf(pdf_ok, max_pages=100).page_count == 100


def test_mixed_scan_detected(tmp_path: Path) -> None:
    pages = ["Readable text page with enough characters here."] * 5 + [""] * 5
    pdf = _write(tmp_path, "mixed.pdf", make_text_pdf(pages))
    with pytest.raises(OcrRequiredError) as excinfo:
        extract_pdf(pdf, max_pages=100)
    assert "pages_with_text=5/10" in (excinfo.value.detail or "")


def test_scan_only_detected(tmp_path: Path) -> None:
    pdf = _write(tmp_path, "scan.pdf", make_text_pdf(["", "", ""]))
    with pytest.raises(OcrRequiredError):
        extract_pdf(pdf, max_pages=100)


def test_encrypted_rejected(tmp_path: Path) -> None:
    pdf = _write(tmp_path, "enc.pdf", make_encrypted_pdf())
    with pytest.raises(PasswordProtectedError):
        extract_pdf(pdf, max_pages=100)


def test_corrupt_rejected(tmp_path: Path) -> None:
    pdf = _write(tmp_path, "bad.pdf", make_corrupt_pdf())
    with pytest.raises(ParseFailedError):
        extract_pdf(pdf, max_pages=100)


# --- pipeline + artifacts ---


def test_pipeline_and_artifacts_roundtrip(tmp_path: Path) -> None:
    doc_dir = tmp_path / "docX"
    doc_dir.mkdir()
    (doc_dir / "original.pdf").write_bytes(make_text_pdf(
        ["1. Subject\nThe supplier delivers goods of proper quality to the buyer.",
         "4.2 Penalties\nThe penalty is limited by the total contract price."]
    ))
    approx = ApproxTokenLength()
    stages: list[str] = []
    result = parse_document(
        doc_dir=doc_dir, content_type="application/pdf", max_pages=100,
        chunk_tokens=60, overlap_tokens=10,
        length_fn=approx, length_fn_id=approx.id, progress=stages.append,
    )
    assert stages == ["parse", "normalize", "split"]
    assert result.page_count == 2
    assert result.normalized_pdf.name == "original.pdf"
    assert result.chunks, "expected at least one chunk"

    artifact = save_parse_artifacts(doc_dir, result, chunk_tokens=60, overlap_tokens=10)
    import json

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    canonical = (doc_dir / "parse" / "canonical.txt").read_text(encoding="utf-8")
    assert payload["length_fn"] == approx.id
    assert payload["page_count"] == 2
    for chunk in payload["chunks"]:
        assert canonical[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        assert 1 <= chunk["page_from"] <= chunk["page_to"] <= 2
