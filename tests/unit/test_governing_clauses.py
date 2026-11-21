from dataclasses import replace

import pytest
from app.inference.contract import RerankDocument
from app.qa.evidence import issue_evidence
from app.qa.grounding import ProvenanceError, validate_claims
from app.qa.schemas import Claim, Evidence
from app.retrieval.clauses import list_rerank_documents, local_list_lead
from app.retrieval.search import ContextChunk

HEADING = "8.3. С письменного согласия заказчика исполнитель вправе:"
ITEM = "8.3.2. Передавать результаты испытаний лаборатории."


def test_reranker_scores_original_candidates_with_their_governing_text() -> None:
    texts = {"2": ITEM, "4": "8.4. Другие условия", "0": HEADING,
             "1": "8.3.1. Привлекать специалистов."}
    documents = [RerankDocument(id=key, text=value) for key, value in texts.items()]
    scored, parents = list_rerank_documents(documents, {key: int(key) for key in texts})
    assert [d.id for d in scored] == list(texts)
    assert parents == {"1": "0", "2": "0"}
    assert scored[0].text == HEADING + "\n" + ITEM
    assert scored[1].text == texts["4"]
    assert scored[2].text.startswith(HEADING + "\n" + ITEM)


@pytest.mark.parametrize("next_text,gap", [
    ("8.30.1. Передавать отчёт.", 1),  # same digits in a different clause
    ("8.3.2. Передавать отчёт.", 2),   # unseen intervening chunk could open a new list
])
def test_unproven_heading_is_never_borrowed(next_text: str, gap: int) -> None:
    docs = [RerankDocument(id="a", text=HEADING), RerankDocument(id="b", text=next_text)]
    assert list_rerank_documents(docs, {"a": 0, "b": gap}) == (docs, {})


def source(key: str, text: str, page: int, start: int) -> ContextChunk:
    return ContextChunk(chunk_id=key, chunk_index=page, text=text, page_from=page,
                        section=None, page_to=page, char_start=start, char_end=start + len(text),
                        rerank_score=.92)


def test_verifier_and_answer_share_both_exact_spans_on_different_pages() -> None:
    heading = source("heading", HEADING, 1, 50)
    child = replace(source("child", ITEM, 2, 161), list_heading_id="heading")
    claim = Claim(text="Передача возможна при письменном согласии заказчика.", evidence=[
        Evidence(chunk_id="s2", quote="Передавать результаты испытаний лаборатории."),
        Evidence(chunk_id="s1", quote=HEADING),
    ])
    cited = validate_claims([claim], {"s1": heading, "s2": child})[0]
    assert [(q.quote_raw, q.page, q.char_start, q.char_end) for q in cited] == [
        (HEADING, 1, 50, 50 + len(HEADING)),
        (claim.evidence[0].quote, 2, 161 + ITEM.index("Передавать"), 161 + len(ITEM)),
    ]
    claim.evidence.pop()
    with pytest.raises(ProvenanceError, match="вводная"):
        validate_claims([claim], {"s2": child})


def test_inline_lead_for_a_numbered_item_ends_at_next_unrelated_clause() -> None:
    text = HEADING + "\n8.3.1. Привлекать специалистов.\n" + ITEM + "\n8.4. Оплата услуг."
    assert local_list_lead(text, text.index(ITEM)) == HEADING
    assert local_list_lead(text, text.index("результаты")) == HEADING
    assert local_list_lead(text, text.index("Оплата")) is None


@pytest.mark.parametrize("decision", ["вправе", "не вправе"])
def test_an_orphan_infinitive_cannot_prove_permission_or_prohibition(decision: str) -> None:
    claim = Claim(text=f"Исполнитель {decision} передавать результаты лаборатории.",
                  evidence=[Evidence(chunk_id="item", quote=ITEM)])
    with pytest.raises(ProvenanceError, match="без вводной фразы"):
        validate_claims([claim], {"item": source("child", ITEM, 1, 0)},
                        question="Вправе ли исполнитель передавать результаты?")


def test_complete_rule_with_its_own_permission_can_stand_alone() -> None:
    quote = "С согласия заказчика допускается передача отчёта."
    item = ITEM + " " + quote
    claim = Claim(text="Отчёт можно передавать с согласия заказчика.",
                  evidence=[Evidence(chunk_id="item", quote=quote)])
    cited = validate_claims([claim], {"item": source("child", item, 2, 161)},
                            question="Можно ли передавать отчёт?")
    assert len(cited[0]) == 1


def test_general_and_specific_activities_remain_visible_in_inclusive_answer() -> None:
    heading = source("heading", "3.4. Заказчику запрещено:", 1, 0)
    first = replace(source("a", "3.4.1. Передавать материалы третьим лицам.", 1, 60),
                    list_heading_id="heading")
    second = replace(source("b", "3.4.3. Передавать фотографии через сеть.", 2, 220),
                     list_heading_id="heading")
    sources = {"lead": heading, "first": first, "second": second}
    claim = Claim(text="Передача фотографий через сеть запрещена.",
                  evidence=["second.1", "first.1"])
    validate_claims([claim], sources, spans=issue_evidence(sources), render_conditional_rules=True,
                    question="Можно ли передавать материалы, в том числе фотографии через сеть?")
    assert claim.text.splitlines() == [heading.text, first.text, second.text]
