from types import SimpleNamespace

from app.qa.evidence import issue_evidence
from app.qa.grounding import qualify_numeric_claims, validate_claims
from app.qa.schemas import Claim, DraftAnswer
from app.qa.source_details import complete_source_details
from app.retrieval.offer_details import matched_tariff, prioritize_source_details
from app.retrieval.search import ContextChunk
from app.retrieval.tables import attach_price_headings


def chunk(i, text):
    return ContextChunk(str(i), i, text, None, 1, 1, 1000 * i, 1000 * i + len(text), 1)


def context(chunks):
    units = {c.chunk_id: [c] for c in chunks}
    attach_price_headings(units)
    return {f"source-{i}": unit[-1] for i, unit in units.items()}


def run(sources, question, drafts=()):
    spans = issue_evidence(sources)
    previous = [
        Claim(text=text, evidence=[next(k for k, q in spans.items() if q.chunk.chunk_id == str(i))])
        for text, i in drafts
    ]
    draft = DraftAnswer(answerable=bool(previous), missing_information=[], claims=previous)
    result = complete_source_details(question, draft, spans)
    refs = validate_claims(result.claims, sources, question=question, spans=spans)
    qualify_numeric_claims(result.claims, refs)
    return result, refs


def test_signed_price_column_keeps_the_product_and_quantity_when_draft_refused():
    head = chunk(0, "Северный филиал\nНаименование Ед. изм. Цена, руб.")
    rows = [
        chunk(1, "1 Практикум (детская группа) разовое занятие шт. 200"),
        chunk(2, "2 Практикум (детская группа) абонемент на 6 занятий шт. 950"),
    ]
    sources = context([head, *rows])
    q = "Сколько стоит абонемент детской группы практикума и на сколько занятий он рассчитан?"
    result, proof = run(sources, q)
    assert result.answerable and "6 занятий" in result.claims[0].text
    assert "950 руб." in result.claims[0].text
    assert [v.chunk.chunk_index for v in proof[0]] == [0, 2]
    result, _ = run(context([*rows]), q)
    assert not result.answerable


def test_marked_price_points_to_its_own_footnote_and_table():
    first = chunk(0, "Западное отделение\n№ Наименование Ед. изм. Цена, руб.")
    price = chunk(1, "1 Обслуживание (дополнение к группе) ** чел. 180")
    split = chunk(
        2,
        "2 Входной билет шт. 400\n** Допуск посетителей при опоздании на десять "
        "минут.\nВосточное отделение\n№ Наименование Ед. изм. Цена, руб.",
    )
    twin = chunk(3, "1 Обслуживание (дополнение к группе) ** чел. 260")
    tail = chunk(4, "2 Входной билет шт. 550\n** Допуск посетителей при опоздании на три минуты.")
    sources = context([first, price, split, twin, tail])
    q = "Что стоит допуск посетителей при опоздании в западном отделении и через сколько минут?"
    result, proof = run(sources, q)
    assert ["180 руб. за человека" in c.text for c in result.claims] == [True, False]
    assert "десять минут" in result.claims[1].text
    assert [v.chunk.chunk_index for v in proof[0]] == [0, 1]
    assert proof[1][0].chunk.chunk_index == 2
    assert (
        matched_tariff(q.replace("западном отделении", "отделении"), list(sources.values())) is None
    )
    assert matched_tariff(q.replace("Что стоит", "Сравните цены"), list(sources.values())) is None


def test_combined_recurring_schedule_draws_both_answers_from_one_passage():
    text = (
        "Каждый четверг в клубе проходит открытая тренировка (посещение по пропуску).\n"
        "Начало тренировки в 18.30."
    )
    q = "Когда начинается открытая тренировка в клубе и нужен ли пропуск?"
    result, proof = run(
        context([chunk(0, text)]), q, [("Открытая тренировка начинается в 18.30.", 0)]
    )
    assert "Каждый четверг" in result.claims[0].text and "по пропуску" in result.claims[0].text
    assert proof[0][0].quote_raw == text


def test_frequency_moves_the_adjacent_matching_clause_into_context_and_answer():
    base = chunk(8, "5.1. Выдача сезонного пропуска приостанавливается на 8 дней.")
    rule = chunk(9, "5.2. Приостановка выдачи сезонного пропуска допускается два раза за сезон.")
    different = chunk(10, "7.1. Приостановка выдачи сезонного пропуска допускается три раза.")
    ordered = [SimpleNamespace(id=c.chunk_id) for c in (base, different, rule)]
    units = {c.chunk_id: [c] for c in (base, rule, different)}
    q = "На сколько дней и сколько раз возможна приостановка выдачи сезонного пропуска?"
    assert [p.id for p in prioritize_source_details(q, ordered, units)] == ["8", "9", "10"]
    result, _ = run(
        context([base, rule, different]),
        q,
        [("Выдача сезонного пропуска приостанавливается на 8 дней.", 8)],
    )
    assert len(result.claims) == 2 and "два раза за сезон" in result.claims[1].text


def test_ticket_includes_what_the_purchased_ticket_actually_entitles_the_holder_to():
    raw = (
        "3.2. Держатель вносит депозит и покупает один талон, дающий право на однократный "
        "вход в зону обслуживания."
    )
    q = "Какие требования предъявлены к депозиту и талону при первом визите?"
    result, refs = run(
        context([chunk(4, raw)]), q, [("Держатель вносит депозит и покупает один талон.", 4)]
    )
    assert "однократный вход" in result.claims[0].text and refs[0][0].quote_raw == raw


def test_maturity_keeps_both_return_procedure_and_the_actual_trigger():
    rule = chunk(
        4,
        "6.3. Пропуск выдаётся на срок не более 3 лет и переходит к держателю "
        "при невозврате по истечении указанного срока в порядке, установленном "
        "пунктом 7, по цене залога.",
    )
    trigger = chunk(
        5,
        "6.3.1. Невозврат пропуска в течение 3 лет считается согласием держателя "
        "приобрести пропуск по цене залога.",
    )
    q = "Когда невозврат пропуска держателем означает приобретение и какая цена предусмотрена?"
    result, _ = run(
        context([rule, trigger]),
        q,
        [("Пропуск переходит к держателю при невозврате по истечении срока по цене залога.", 4)],
    )
    assert "пунктом 7" in result.claims[0].text
    assert "в течение 3 лет" in result.claims[1].text
