from dataclasses import replace

import pytest
from app.qa.evidence import issue_evidence
from app.qa.grounding import quote_at_offsets, validate_claims
from app.qa.numbers import (
    clause_reference_heading,
    numeric_evidence_texts,
    qualify_numeric_text,
    unsupported_numbers,
)
from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk


def _chunk(chunk_id, text, start=0, heading_id=None):
    return ContextChunk(chunk_id, 0, text, None, 1, 1, start, start + len(text), 0.9,
                        list_heading_id=heading_id)


def test_document_number_is_supported_only_as_the_same_instrument_type_and_id():
    text = "Договор № 23-А\nСтатья 7 устанавливает срок 4 дня."
    assert not unsupported_numbers("По договору №23-А применяется статья 7.", [text], "")
    for claim in ("Сумма — 23 рубля.", "Протокол №23-А действует.",
                  "Статья 4 устанавливает срок.", "Договор №23 действует.",
                  "В договоре №72-А установлен срок."):
        assert unsupported_numbers(claim, [text], "Что установлено договором №72-А?")
    # A document number cannot be borrowed from an article or a duration.
    assert unsupported_numbers("По протоколу № 7 установлен срок.", [text], "")


def test_standalone_instrument_title_gets_exact_evidence_but_contents_entry_does_not():
    text = ("Указатель\nДоговор № 8 .......... 47\nКонец другого документа.\n"
            "Договор № 8\nо порядке расчётов")
    spans = issue_evidence({"source-1": _chunk("a", text, start=100)})
    title, = [q for q in spans.values() if q.quote_raw == "Договор № 8"]
    assert title.char_start == 100 + text.rindex("Договор № 8")
    assert title.char_end == title.char_start + len(title.quote_raw)


def test_referenced_local_clause_uses_only_its_nearest_actual_lead():
    text = ("6.7. Поставщик возмещает:\nа) стоимость замены;\nб) расходы на доставку.\n"
            "6.8. Покупатель возмещает:\nа) расходы на хранение.")
    claim = "Согласно пункту 6.7 возмещаются расходы на доставку."
    interval = clause_reference_heading(claim, text, text.index("б) расходы"))
    assert interval and text[slice(*interval)] == "6.7. Поставщик возмещает:"
    assert not clause_reference_heading(claim, text, text.index("а) расходы на хранение"))
    assert not clause_reference_heading(claim.replace("6.7", "9.1"), text, text.index("б) расходы"))
    sources = {"source-1": _chunk("section", text)}
    spans = issue_evidence(sources)
    key = next(key for key, span in spans.items() if span.quote_raw.startswith("б) расходы"))
    accepted = validate_claims([Claim(text=claim, evidence=[key])], sources, spans=spans)[0]
    assert accepted[0].quote_raw == "6.7. Поставщик возмещает:"
    assert accepted[1].quote_raw.startswith("б) расходы на доставку.")


def test_currency_requires_one_complete_row_and_its_own_heading():
    head = _chunk("header", "№ Позиция Ед. изм. Цена, руб-")
    row = _chunk("row", "6 Комплект на 8 посещений шт. 1 870", 60, head.chunk_id)
    other = _chunk("other", "7 Комплект на 9 посещений шт. 2100", 150, head.chunk_id)

    def quote(chunk):
        return quote_at_offsets(chunk, 0, len(chunk.text))

    claim = "Комплект на 8 посещений стоит 1870 рублей."
    texts = numeric_evidence_texts([quote(head), quote(row)])
    assert not unsupported_numbers(claim, texts, "")
    sources = {"source-1": head, "source-2": row}
    accepted = validate_claims([Claim(text=claim, evidence=["source-2.1"])], sources,
                               spans=issue_evidence(sources))[0]
    assert [(q.char_start, q.quote_raw) for q in accepted] == [(0, head.text), (60, row.text)]
    assert unsupported_numbers("Комплект стоит 8 рублей.", texts, "")
    for evidence in ([quote(row)], [quote(head), quote(replace(row, list_heading_id="foreign"))],
                     [quote(head), quote_at_offsets(row, 5, len(row.text))],
                     [quote(head), quote(other), quote(row)]):
        assert unsupported_numbers(claim, numeric_evidence_texts(evidence), "")


def test_expiry_and_maximum_duration_need_separate_support():
    limit = "Оборудование передают на срок не\nболее 4 (четырех) месяцев."
    expiry = "По истечении 4 месяцев с даты выдачи оператор начисляет плату."
    claim = "Плата начисляется через 4 месяца с даты выдачи."
    assert qualify_numeric_text(claim, [limit, expiry]) == claim
    bounded = "Срок передачи — не более 4 месяцев."
    assert qualify_numeric_text(bounded, [limit, expiry]) == bounded
    with pytest.raises(ValueError, match="отсч"):
        qualify_numeric_text(claim, [limit])
    with pytest.raises(ValueError, match="отсч"):
        qualify_numeric_text(claim, [limit, "Проверка идёт в течение 4 месяцев."])
    with pytest.raises(ValueError, match="отсч"):
        qualify_numeric_text(claim, ["Проверка идёт в течение 4 месяцев."])
    with pytest.raises(ValueError, match="разные условия"):
        qualify_numeric_text("Срок 4 месяца.", [limit, expiry])
