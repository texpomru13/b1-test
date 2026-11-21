"""Regressions for qualifications, legal definitions and grounded rereading."""

from types import SimpleNamespace

from app.qa.completeness import add_defined_day, preserve_qualifications
from app.qa.extractive import from_selection, reading_passages
from app.qa.grounding import quote_at_offsets
from app.qa.schemas import Claim
from app.retrieval.day_definition import prioritize_day_definition
from app.retrieval.search import ContextChunk


def chunk(text: str, key: str = "a", scope: str | None = None) -> ContextChunk:
    return ContextChunk(chunk_id=key, chunk_index=0, text=text, section=None, page_from=1,
                        page_to=1, char_start=0, char_end=len(text), rerank_score=0, scope_id=scope)


def test_answer_keeps_the_applicable_clause_and_its_exception() -> None:
    source = chunk("Плата за хранение начисляется со дня передачи ящика, "
                   "за исключением периода ремонта. Клиент открывает ящик своим ключом.")
    claim = Claim(text="Плата за хранение начисляется.", evidence=["source-1.1"])
    preserve_qualifications([claim], {0: [quote_at_offsets(source, 0, len(source.text))]})
    assert "со дня передачи" in claim.text and "за исключением периода ремонта" in claim.text
    assert "ключом" not in claim.text


def test_condition_is_bound_to_its_subject_and_quantity() -> None:
    source = chunk("Комиссия за обработку составляет 4 процента. "
                   "Плата за хранение начисляется со дня передачи ящика.")
    claim = Claim(text="Комиссия составляет 4 процента.", evidence=["source-1.1"])
    preserve_qualifications([claim], {0: [quote_at_offsets(source, 0, len(source.text))]})
    assert claim.text == "Комиссия составляет 4 процента."


def test_link_day_definition_to_bare_period_only_in_its_scope() -> None:
    source = chunk("Заявка рассматривается в течение 7 дней.", scope="a")
    definition = chunk("День, дни — рабочие дни.", "b")
    other = chunk("День, дни — календарные дни.", "c", scope="c")
    claim = Claim(text=source.text, evidence=["source-1.1"])
    quoted = [quote_at_offsets(source, 0, len(source.text))]
    add_defined_day(claim, quoted, [other])
    assert len(quoted) == 1
    add_defined_day(claim, quoted, [other, definition])
    assert "рабочие дни" in claim.text and quoted[-1].chunk is definition
    # A local explicit calendar-day deadline takes precedence over a default.
    explicit = chunk("Заявка рассматривается в течение 7 календарных дней.")
    c = Claim(text=explicit.text, evidence=["source-1.1"])
    q = [quote_at_offsets(explicit, 0, len(explicit.text))]
    add_defined_day(c, q, [definition])
    assert len(q) == 1 and c.text == explicit.text


def test_rank_formal_definition_alongside_direct_deadline_without_picking_other_usage() -> None:
    points = [SimpleNamespace(id=str(i), payload={"text": t}) for i, t in enumerate(
        ["Поставка занимает 7 дней.", "Отдел работает в рабочие дни.", "День, дни — рабочие дни."])]
    assert prioritize_day_definition("В какой срок поставят товар?", points) == [
        points[0], points[2], points[1]]
    assert prioritize_day_definition("Какой цвет товара?", points) == points


def test_second_read_is_anchored_to_evidence_or_skipped() -> None:
    c = chunk("Сотрудник представляет заявление о возмещении расходов в бумажной форме.")
    s = {"source-1.1": quote_at_offsets(c, 0, len(c.text))}
    assert "source-1.1" in reading_passages("Как сотрудник представляет заявление?", s)
    assert not reading_passages("Какова температура плавления?", s)


def test_extractive_repair_answers_yes_no_only_for_a_closed_question() -> None:
    c = chunk("Ключ выдаётся только держателю действующего пропуска.")
    choices = {"source-1.1": quote_at_offsets(c, 0, len(c.text))}
    raw = {"passages": ["source-1.1"], "polarity": "no"}
    closed = from_selection(raw, choices, "Можно ли получить ключ без пропуска?")
    assert closed.claims[0].text == "Нет. " + c.text
    assert closed.claims[0].evidence == ["source-1.1"]
    assert from_selection(raw, choices, "Кому выдают ключ?").claims[0].text == c.text
    empty = from_selection({"passages": [], "polarity": "no"}, choices, "Можно ли взять ключ?")
    assert not empty.answerable and not empty.claims
