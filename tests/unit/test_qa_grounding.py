import pytest
from app.qa.grounding import ProvenanceError, locate_quote, validate_claims
from app.qa.schemas import Claim, Evidence
from app.retrieval.search import ContextChunk

CHUNK_TEXT = (
    "5.1 За нарушение сроков поставки Поставщик уплачивает Покупателю\n"
    "неустойку в размере 0,1% от стоимости товара за каждый день просрочки,\n"
    "но не более 10% от общей цены договора в течение десяти рабочих дней."
)


def chunk(chunk_id: str = "source-1", *, char_start: int = 100,
          segments: list | None = None) -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id, chunk_index=3, text=CHUNK_TEXT, section="5. Ответственность",
        page_from=2, page_to=3, char_start=char_start,
        char_end=char_start + len(CHUNK_TEXT), rerank_score=0.9,
        segments=segments or [],
    )


def test_quote_located_across_line_breaks_with_offsets() -> None:
    c = chunk()
    located = locate_quote(c, "уплачивает Покупателю неустойку в размере 0,1%")
    assert "\n" in located.quote_raw  # raw span preserves the original newline
    assert CHUNK_TEXT[located.char_start - 100 : located.char_end - 100] == located.quote_raw


def test_fabricated_quote_rejected() -> None:
    with pytest.raises(ProvenanceError, match="not a substring"):
        locate_quote(chunk(), "неустойка составляет 0,5% в день")


def test_bracketed_source_id_is_accepted_as_faithful_echo() -> None:
    claims = [Claim(
        text="Неустойка — 0,1% в день.",
        evidence=[Evidence(chunk_id="[source-1]",
                           quote="неустойку в размере 0,1% от стоимости товара")],
    )]
    located = validate_claims(claims, {"source-1": chunk()})
    assert located[0][0].quote_raw.startswith("неустойку")


def test_fabricated_chunk_id_rejected() -> None:
    claims = [Claim(text="Неустойка 0,1% в день.",
                    evidence=[Evidence(chunk_id="source-99", quote="0,1%")])]
    with pytest.raises(ProvenanceError, match="not in the allowed context"):
        validate_claims(claims, {"source-1": chunk()})


def test_numbers_must_come_from_evidence() -> None:
    claims = [Claim(
        text="Неустойка составляет 0,3% в день.",  # number not in the quote
        evidence=[Evidence(chunk_id="source-1",
                           quote="неустойку в размере 0,1% от стоимости товара")],
    )]
    with pytest.raises(ProvenanceError, match=r"0\.3"):
        validate_claims(claims, {"source-1": chunk()})


def test_day_kind_must_match_evidence() -> None:
    claims = [Claim(
        text="Срок составляет десять календарных дней.",
        evidence=[Evidence(chunk_id="source-1",
                           quote="в течение десяти рабочих дней")],
    )]
    with pytest.raises(ProvenanceError, match="календарн"):
        validate_claims(claims, {"source-1": chunk()})


def test_valid_claim_passes_with_located_quotes() -> None:
    claims = [Claim(
        text="Неустойка — 0,1% в день, но не более 10% цены договора.",
        evidence=[Evidence(
            chunk_id="source-1",
            quote="неустойку в размере 0,1% от стоимости товара за каждый день "
                  "просрочки, но не более 10% от общей цены договора",
        )],
    )]
    located = validate_claims(claims, {"source-1": chunk()})
    assert located[0][0].page == 2  # single-segment fallback: page_from


def test_quote_page_resolved_from_segments() -> None:
    # chunk spans pages 2-3; the segment map places the tail on page 3
    seg = [
        {"page": 2, "char_start": 100, "char_end": 100 + 120},
        {"page": 3, "char_start": 100 + 120, "char_end": 100 + len(CHUNK_TEXT)},
    ]
    c = chunk(segments=seg)
    located = locate_quote(c, "десяти рабочих дней")
    assert located.page == 3 and located.page_to == 3
