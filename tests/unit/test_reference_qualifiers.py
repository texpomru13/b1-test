import pytest
from app.qa.completeness import preserve_qualifications
from app.qa.glosses import without_unproven_gloss
from app.qa.grounding import ProvenanceError, locate_quote, remove_repeated_claims
from app.qa.relevance import refine_answer_scope
from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk


def assess(question, rows):
    sources = {}
    claims = []
    located = {}
    for i, (draft, source) in enumerate(rows):
        key = f"source-{i + 1}"
        chunk = ContextChunk(key, i, source, None, 1, 1, i * 1000, i * 1000 + len(source), 1)
        sources[key] = chunk
        claims.append(Claim(text=draft, evidence=[key + ".1"]))
        located[i] = [locate_quote(chunk, source)]
    preserve_qualifications(claims, located)
    located = refine_answer_scope(question, claims, located, sources)
    remove_repeated_claims(claims, located)
    return [c.text for c in claims]


@pytest.mark.parametrize("gloss", [" (реестр комплектации)", ", то есть реестр комплектации"])
def test_neighboring_name_is_not_an_expansion(gloss):
    q = "Где закрепить результат проверки?"
    quote = "Результат проверки записывают в РК. Реестр комплектации ведёт другая организация."
    assert assess(q, [("Результат проверки записывают в РК" + gloss + ".", quote)]) == [
        "Результат проверки записывают в РК."]
    with pytest.raises(ProvenanceError, match="определением"):
        assess("Что означает РК?", [("Запись в РК" + gloss + ".", quote)])


@pytest.mark.parametrize("proof", [
    "РК — реестр контроля. Запись вносится в РК.",
    "Реестр контроля (далее — РК) ведётся по установленной форме.",
])
def test_printed_definition_can_be_given(proof):
    statement = "Запись вносится в РК (реестре контроля)."
    assert without_unproven_gloss(statement, [proof], "Что означает РК?") == statement


def test_reference_does_not_authorize_new_types_of_event():
    q = "Какой общий и разовый предел установлен согласно п. 3.7?"
    source = ("3.7. Платежи, указанные в подпунктах а) и в), не превышают 9% общего резерва, "
              "а по одному событию — 6%, если условиями сторон не установлено иное.")
    draft = ("Платежи из п. 3.7 (по происшедшим и предполагаемым случаям) "
             "не превышают 9% резерва, по одному событию — 6%, если не установлено иное.")
    answer, = assess(q, [(draft, source)])
    assert "предполагаем" not in answer and "происшедшим" not in answer
    assert all(term in answer for term in ("9%", "6%", "не установлено иное"))


def test_closing_time_attaches_to_the_cited_activity_and_date():
    record = ("Мастер ежедневно регистрирует наблюдения до их окончательного завершения "
              "с сохранением протоколов по итогам смены в журнале, доступном комиссии.")
    extra = "Пропуск оформляют по окончании квартала в бюро."
    answers = assess("Где мастер сохраняет наблюдения и когда?", [
        ("Мастер ежедневно регистрирует наблюдения.", record),
        ("Мастер сохраняет протоколы в журнале.", record),
    ])
    assert any("по итогам смены" in a for a in answers)
    assert all("квартала" not in a for a in assess(
        "Где хранятся документы?", [("Мастер регистрирует наблюдения.", record + " " + extra)]))


@pytest.mark.parametrize("reverse", [False, True])
def test_change_of_a_different_attribute_is_excluded_unless_explicitly_requested(reverse):
    q = "Как оформляют изменение срока перевозки?"
    term = ("Срок перевозки может быть изменён соглашением.", "Срок перевозки может быть "
            "изменён соглашением.")
    fare = ("При изменении тарифа перевозки стороны составляют протокол.",
            "При изменении тарифа перевозки стороны составляют протокол.")
    rows = [term, fare] if reverse else [fare, term]
    assert assess(q, rows) == [term[0]]
    assert len(assess(q + " Как оформляют изменение тарифа перевозки?", rows)) == 2
    different_question = "Как оформляют изменение тарифа перевозки?"
    direct = ("Тариф перевозки может быть изменён соглашением.",
              "Тариф перевозки может быть изменён соглашением.")
    other = ("При изменении срока перевозки стороны составляют протокол.",
             "При изменении срока перевозки стороны составляют протокол.")
    assert assess(different_question, [other, direct]) == [direct[0]]


def test_change_with_both_properties_and_a_different_object_remains_verifiable():
    q = "Как оформляют изменение срока перевозки?"
    direct = ("Срок перевозки может быть изменён соглашением.",
              "Срок перевозки может быть изменён соглашением.")
    linked = ("При изменении стоимости перевозки срок перевозки пересчитывают.",
              "При изменении стоимости перевозки срок перевозки пересчитывают.")
    assert len(assess(q, [direct, linked])) == 2


def test_reverse_expansion_is_restored_from_the_exact_operative_rule():
    quote = "Комиссия согласовывает новый порядок путем указания об этом в ПР."
    wrong = "Комиссия согласовывает новый порядок путем указания об этом в Паспорте ремонта (ПР)."
    assert assess("Как оформить согласование нового порядка?", [(wrong, quote)]) == [quote]
    correct = "Комиссия закрепляет согласие в Паспорте ремонта (ПР)."
    defined = quote + " Паспорт ремонта (далее — ПР) подписывается комиссией."
    assert without_unproven_gloss(correct, [defined], "Где указать согласие?") == correct
    with pytest.raises(ProvenanceError, match="раскрытое название"):
        assess("Как расшифровать ПР?", [(wrong, quote)])
