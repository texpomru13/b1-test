import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from app.ingestion import ocr, pdf_loader
from app.ingestion.chunker import ApproxTokenLength
from app.ingestion.errors import OcrRequiredError
from app.ingestion.pdf_loader import extract_pdf
from app.ingestion.pipeline import CONTENT_TYPE_PDF, parse_document, save_parse_artifacts
from app.jobs.indexer import _chunk_payload
from app.qa.assemble import assemble_answer
from app.qa.evidence import issue_evidence
from app.qa.grounding import validate_claims
from app.qa.schemas import Claim, DraftAnswer
from app.qa.workflow import QaResult
from app.retrieval.search import Diagnostics, RetrievalResult, _context_unit
from pypdf import PdfReader, PdfWriter

DOCUMENTS = Path(__file__).resolve().parents[2] / "test_documents"
SCAN = next(DOCUMENTS.glob("06_*.pdf"))
NATIVE = next(DOCUMENTS.glob("15_*.pdf"))
# Transcribed from physical page 3 of fixture 06. Its printed folio remains 3
# when composed into physical page 2 of this mixed PDF.
OCR_SENTENCE = (
    "В случае отсутствия в течение 5 (пяти) календарных дней со дня акцепта настоящей оферты "
    "у Лицензиата к Лицензиару письменных претензий/требований по вопросу предоставления прав "
    "использования ПО, права использования ПО по Договору признаются предоставленными надлежащим "
    "образом в соответствии с условиями Договора."
)


def test_real_mixed_pdf_keeps_native_text_and_cites_ocr_on_its_physical_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = PdfWriter()
    native = PdfReader(NATIVE).pages[0]
    scan = PdfReader(SCAN).pages[2]
    assert native.extract_text().strip() and not scan.extract_text().strip()
    writer.add_page(native)
    writer.add_page(scan)
    path = tmp_path / "original.pdf"
    writer.write(path)

    recognize = Mock(return_value=(
        "5.1. С момента акцепта настоящей оферты право использования ПО считается "
        "предоставленным Лицензиату. " + OCR_SENTENCE
    ))
    monkeypatch.setattr(pdf_loader, "recognize_page", recognize)
    parsed = parse_document(doc_dir=tmp_path, content_type=CONTENT_TYPE_PDF, max_pages=2,
                            chunk_tokens=450, overlap_tokens=64, length_fn=ApproxTokenLength(),
                            length_fn_id=ApproxTokenLength.id)
    recognize.assert_called_once_with(path, 2)
    artifact = json.loads(save_parse_artifacts(tmp_path, parsed, chunk_tokens=450,
                                               overlap_tokens=64).read_text())
    assert [p["ocr"] for p in artifact["pages"]] == [False, True]
    source = parsed.canonical.text
    assert source.startswith(native.extract_text().strip().splitlines()[0].strip())
    raw = next(c for c in artifact["chunks"] if OCR_SENTENCE in c["text"])
    payload = _chunk_payload(raw, job={"doc_id": "mixed"}, run_id="run", fingerprint="fp",
                             page_count=2, chunk_count=len(artifact["chunks"]), created_at="now",
                             chunks=artifact["chunks"])
    chunk = _context_unit("point", payload)[-1]
    sources = {"source-1": chunk}
    spans = issue_evidence(sources)
    ref = next(key for key, span in spans.items() if OCR_SENTENCE in span.quote_raw)
    claim = Claim(text=OCR_SENTENCE, evidence=[ref])
    state = QaResult(RetrievalResult([chunk], [chunk.chunk_id], Diagnostics()),
                     draft=DraftAnswer(answerable=True, claims=[claim], missing_information=[]),
                     located=validate_claims([claim], sources, spans=spans))
    citation = assemble_answer(state).citations[0]
    assert citation.page == citation.page_to == 2 and citation.ocr_pages == [2]
    assert source[citation.char_start:citation.char_end] == citation.text == OCR_SENTENCE


@pytest.mark.parametrize("result", [FileNotFoundError("tesseract unavailable"),
                                    subprocess.CompletedProcess([], 0, "eng\nosd\n", "")])
def test_ocr_requires_installed_binaries_and_both_language_packs(
    monkeypatch: pytest.MonkeyPatch, result: object,
) -> None:
    ocr._check_languages.cache_clear()
    command = Mock(return_value=result)
    if isinstance(result, Exception):
        command.side_effect = result
    monkeypatch.setattr(ocr.subprocess, "run", command)
    with pytest.raises(OcrRequiredError):
        ocr.recognize_page(SCAN, 1)


def test_missing_page_renderer_is_a_visible_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocr, "_check_languages", lambda: None)
    monkeypatch.setattr(ocr.subprocess, "run", Mock(side_effect=FileNotFoundError("pdftoppm")))
    with pytest.raises(OcrRequiredError, match="physical page 3"):
        ocr.recognize_page(SCAN, 3)


def test_unreadable_image_document_fails_without_an_empty_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = PdfWriter()
    writer.add_page(PdfReader(SCAN).pages[2])
    path = tmp_path / "original.pdf"
    writer.write(path)
    monkeypatch.setattr(pdf_loader, "recognize_page", lambda *_: " . — ")
    with pytest.raises(OcrRequiredError, match="no readable text"):
        extract_pdf(path, max_pages=1)
