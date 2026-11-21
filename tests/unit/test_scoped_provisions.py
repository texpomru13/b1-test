import pytest
from app.qa.evidence import issue_evidence
from app.qa.extractive import bind_local_clauses, from_selection
from app.qa.schemas import Claim, DraftAnswer, is_incomplete_draft
from app.retrieval.articles import article_units
from pydantic import ValidationError

from tests.unit.test_article_context import sources


def test_party_defined_by_an_article_can_be_granted_a_right_elsewhere():
    units = sources(["Статья 8\n1. Участниками признаются лаборатории.",
                     "Статья 9\n1. Иным организациям предоставляется заверенное уведомление."],
                    [0, 1])
    assert article_units(
        "Может ли организация, которая не относится к участникам из статьи 8, "
        "потребовать уведомление?", units,
    ) is None
    assert article_units("Какие участники указаны в статье 8?", units) == {"0"}


def test_short_roman_articles_pack_only_their_own_contiguous_paragraphs():
    texts = ["Статья III", "1. План составляет отдел.", "2. Доступ закрытый.\nСтатья IV",
             "Статья III", "1. Учёт ведёт инспекция.", "2. Доступ открытый.\nСтатья IV"]
    units = sources(texts, [4, 1, 0, 5, 2, 3])
    assert article_units("Кто ведёт учёт и каков доступ по статье III?", units) is None
    assert {q.text for q in units["4"]} == {
        "Статья III", "1. Учёт ведёт инспекция.", "2. Доступ открытый.",
    }
    assert "2. Доступ закрытый." in [q.text for q in units["1"]]
    broken = sources(texts, [0, 1, 2, 3, 5])
    article_units("Кто ведёт учёт по статье III?", broken)
    assert len(broken["5"]) == 1


def test_partial_answers_are_distinct_from_structural_output_errors():
    claim = {"text": "Сведения направляют оператору.", "evidence": ["source-1.1"]}
    for claims, gaps, partial in [([claim], ["Не указана форма уведомления."], True),
                                  ([], ["Не указана форма уведомления."], False),
                                  ([dict(claim, evidence=[])], ["Срок"], False)]:
        with pytest.raises(ValidationError) as exc:
            DraftAnswer.model_validate({"claims": claims, "answerable": True,
                                        "missing_information": gaps})
        assert is_incomplete_draft(exc.value) is partial


def test_foreign_alphabet_sibling_is_not_attached_to_a_nested_roman_list():
    source = ("а) срок выдачи;\n b) параметры проверки, включая:\n"
              "i) скорость,\nii) объём,\n с) адрес.")
    unit = sources([source], [0])["0"][0]
    spans = issue_evidence({"source-1": unit})
    key = next(k for k, q in spans.items() if "параметры" in q.quote_raw)
    assert spans[key].quote_raw == "b) параметры проверки, включая:\ni) скорость,\nii) объём,"
    draft = from_selection({"passages": [key], "polarity": "rule"}, spans, "Какие параметры?")
    assert draft.claims[0].text == "b) параметры проверки, включая: i) скорость, ii) объём"


def test_reordered_rule_may_rebind_only_to_a_matching_clause_of_the_same_source():
    texts = ["(1) Результаты работы представляются голосованием.\n"
             "(2) Пересмотренные оценки в расчёт не принимаются.",
             "(1) Внеплановые оценки в расчёт не принимаются."]
    units = sources(texts, [1, 0])
    spans = issue_evidence({k: v[-1] for k, v in units.items()})
    first = next(k for k, q in spans.items() if "Результаты" in q.quote_raw)
    second = next(k for k, q in spans.items() if "Пересмотренные" in q.quote_raw)
    ready = Claim(text="В расчёт не принимаются пересмотренные оценки.", evidence=[first])
    bind_local_clauses("Как считает комиссия?", [ready], spans)
    assert ready.evidence == [second] and ready.text.startswith("Пересмотренные оценки")
    for text in ("В расчёт принимаются пересмотренные оценки.",
                 "В расчёт не принимаются внеплановые оценки."):
        unchanged = Claim(text=text, evidence=[first])
        bind_local_clauses("Как считает комиссия?", [unchanged], spans)
        assert unchanged.evidence == [first] and unchanged.text == text


def test_definition_example_retains_its_general_category_and_existing_conditions():
    unit = sources(["Данное понятие включает средства малой механизации, включая дрели."],
                   [0])["0"][0]
    spans = issue_evidence({"source-1": unit})
    question = "Охватывает ли это понятие дрели?"
    claim = Claim(text="По правилам договора понятие охватывает дрели.", evidence=["source-1.1"])
    bind_local_clauses(question, [claim], spans)
    assert claim.text.startswith("По правилам договора")
    assert "средства малой механизации" in claim.text
    other = Claim(text="Понятие охватывает шкафы.", evidence=["source-1.1"])
    bind_local_clauses("Охватывает ли понятие шкафы?", [other], spans)
    assert other.text == "Понятие охватывает шкафы."
