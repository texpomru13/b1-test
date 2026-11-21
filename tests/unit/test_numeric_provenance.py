from dataclasses import replace

import pytest
from app.qa.evidence import issue_evidence
from app.qa.grounding import ProvenanceError, quote_at_offsets, validate_claims
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
    def price_row(key, text, start):
        return replace(_chunk(key, text, start, head.chunk_id),
                       table_heading_span=(head.char_start, head.char_end))

    row = price_row("row", "6 Комплект на 8 посещений шт. 1 870", 60)
    other = price_row("other", "7 Комплект на 9 посещений шт. 2100", 150)

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


def test_wrapped_price_header_requires_its_own_rightmost_column_and_citation():
    header = "№ Единица Цена,\nп/п Наименование услуги измерения руб."
    item = "7 Комплект на 8 посещений шт. 730"
    title = _chunk("head", header)
    row = replace(_chunk("item", item, 100, title.chunk_id),
                  table_heading_span=(0, len(header)))
    sources = {"source-1": title, "source-2": row}
    claim = Claim(text="Комплект на 8 посещений стоит 730 рублей.", evidence=["source-2.1"])
    accepted = validate_claims([claim], sources, spans=issue_evidence(sources))[0]
    assert [q.quote_raw for q in accepted] == [header, item]
    assert unsupported_numbers("Комплект стоит 8 рублей.", numeric_evidence_texts(accepted), "")

    # A currency below the price in the middle column cannot confer that
    # currency on the terminal quantity. An unrelated tariff cannot supply it either.
    ambiguous = ["№ Цена, Количество\nп/п Наименование руб. шт.",
                 "№ Единица Цена,\nп/п Наименование Субсидия, руб.",
                 "№ Единица Цена,\nп/п Наименование евро."]
    for other_header in ambiguous:
        assert unsupported_numbers(claim.text, [other_header + "\n" + item], "")
    for other_row in [replace(row, list_heading_id="elsewhere"),
                      replace(row, table_heading_span=(2, len(header))),
                      replace(row, table_heading_span=None)]:
        evidence = [accepted[0], quote_at_offsets(other_row, 0, len(item))]
        assert unsupported_numbers(claim.text, numeric_evidence_texts(evidence), "")


def test_scoped_rule_cites_only_its_actual_instrument_title():
    body = "Решение направляется уполномоченному органу."
    titles = "Договор № 71\nОкончание прежнего текста.\nДоговор № 34"
    parent = _chunk("sections", titles)
    interval = (titles.rindex("Договор № 34"), len(titles))
    child = replace(_chunk("rule", body, len(titles) + 1, parent.chunk_id),
                    scope_heading_span=interval)
    sources = {"source-1": parent, "source-2": child}
    spans = issue_evidence(sources)
    claim = Claim(text="По договору № 34 решение направляют уполномоченному органу.",
                  evidence=["source-2.1"])
    heading, rule = validate_claims([claim], sources, spans=spans)[0]
    assert (heading.quote_raw, heading.char_start, heading.char_end) == ("Договор № 34", *interval)
    assert rule.quote_raw == body
    claim.text = claim.text.replace("34", "71")
    with pytest.raises(ProvenanceError, match="71"):
        validate_claims([claim], sources, question="Кому направить решение по договору №71?",
                        spans=spans)

    # The same heading and the rule can also occupy a single retrieval chunk.
    text = "Решение согласуется отдельно.\nДоговор № 34\n" + body
    left = text.index("Договор")
    local = replace(_chunk("local", text), scope_heading_span=(left, left + len("Договор № 34")))
    sources = {"source-1": local}
    spans = issue_evidence(sources)
    claim.text = "По договору №34 решение направляют уполномоченному органу."
    claim.evidence = [key for key, q in spans.items() if q.quote_raw == body]
    assert [q.quote_raw for q in validate_claims([claim], sources, spans=spans)[0]] == [
        "Договор № 34", body,
    ]
    claim.evidence = [key for key, q in spans.items() if q.quote_raw.startswith("Решение согла")]
    with pytest.raises(ProvenanceError, match="34"):
        validate_claims([claim], sources, spans=spans)


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
