from dataclasses import replace

import pytest
from app.qa.evidence import issue_evidence
from app.qa.grounding import ProvenanceError, validate_claims
from app.qa.numbers import unsupported_numbers
from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk


def test_selected_occurrence_survives_duplicate_sentences_and_page_breaks() -> None:
    text = "Строка повторяется. Строка повторяется.\n\nОплатить\nтовар. Кроме госпошлины."
    chunk = ContextChunk("id", 0, text, None, 1, 2, 100, 100 + len(text), .9,
                         segments=[{"page": 1, "char_start": 100, "char_end": 120},
                                   {"page": 2, "char_start": 120, "char_end": 100 + len(text)}])
    sources = {"source-1": chunk}
    spans = issue_evidence(sources)
    assert len(spans) == 3
    claim = Claim(text="Строка повторяется.", evidence=["source-1.2"])
    selected = validate_claims([claim], sources, spans=spans)[0][0]
    assert selected.char_start == 120 and selected.page == selected.page_to == 2
    assert spans["source-1.3"].quote_raw == "Оплатить\nтовар. Кроме госпошлины."
    other = replace(chunk, chunk_id="outside")
    with pytest.raises(ProvenanceError):
        validate_claims([claim], {"source-1": other}, spans=spans)


def test_numeric_names_section_ids_and_registration_fields_have_separate_roles() -> None:
    quotes = ["13. Реквизиты ООО «Маяк-2». ИНН/КПП 1234567890/123456789, ОГРН 2345678901234."]
    assert not unsupported_numbers("КПП ООО «Маяк-2» составляет 123456789.", quotes, "")
    assert unsupported_numbers("ИНН ООО «Маяк-2» составляет 123456789.", quotes, "")
    assert unsupported_numbers("ООО «Маяк-2» предоставляет 2 единицы.", quotes, "")
    assert unsupported_numbers("У ООО «Маяк-3» ОГРН 2345678901234.", quotes, "")
    clause = "7.5.2. Срок действия составляет 10 дней."
    assert not unsupported_numbers("В п. 7.5.2 установлен срок 10 дней.", [clause], "")
    assert unsupported_numbers("Срок по п. 5.2 составляет 10 месяцев.", [clause], "")
    assert not unsupported_numbers("У акции «4G навсегда» срок 10 дней.", [clause], "",
                                   ("Акция «4G навсегда»",))
    assert unsupported_numbers("Срок составляет 4 дня.", [clause], "", ("Акция «4 дня»",))


def test_premise_conversion_is_never_available_as_the_answer() -> None:
    question = "Что предоставить, если срок годности составляет два года?"
    quote = "Если товар имеет срок годности 18 месяцев и более, остаток составляет год."
    claim = "Если срок годности составляет два года (24 месяца), остаток составляет год."
    assert not unsupported_numbers(claim, [quote], question)
    assert unsupported_numbers(claim.replace("24", "25"), [quote], question)
    assert unsupported_numbers(claim.replace("срок годности", "срок доставки"), [quote], question)
    assert unsupported_numbers("Остаток составляет 24 месяца.", [quote], question)


def test_terminal_price_column_and_currency_cannot_be_borrowed_from_another_row() -> None:
    text = "Оборудование Стоимость, руб. с НДС\nКамера 4G 6 600\nСрок монтажа\n10"
    assert not unsupported_numbers("Цена — 6600,00 рубля.", [text], "")
    assert unsupported_numbers("Цена — 10 рублей.", [text], "")
    assert unsupported_numbers("Срок — 6600 дней.", [text], "")
