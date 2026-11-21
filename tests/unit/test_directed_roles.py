import pytest
from app.qa.grounding import ProvenanceError, validate_claims
from app.qa.roles import reversed_list_actor
from app.qa.schemas import Claim, Evidence
from app.retrieval.search import ContextChunk

RULE = ("Наблюдательный комитет:\n(i) проверяет заявления;\n"
        "(ii) направляет региональному совету заключение о проверке.")


def test_recipient_cannot_become_actor_of_a_supplementary_claim() -> None:
    chunk = ContextChunk("chunk", 0, RULE, None, 1, 1, 0, len(RULE), 1.0, [])
    evidence = [Evidence(chunk_id="source-1", quote=RULE)]
    direct = Claim(text="Наблюдательный комитет проверяет заявления.", evidence=evidence)
    reversed_role = Claim(text="Региональный совет направляет заключение о проверке.",
                          evidence=evidence)
    with pytest.raises(ProvenanceError, match="Адресат действия назван исполнителем"):
        validate_claims([direct, reversed_role], {"source-1": chunk})


@pytest.mark.parametrize("claim", [
    "Наблюдательный комитет направляет региональному совету заключение.",
    "Заключение направляется региональному совету наблюдательным комитетом.",
    "Региональный совет направляет протокол в архив.",
])
def test_distinct_actors_voices_and_actions_remain_verifiable(claim: str) -> None:
    assert reversed_list_actor(claim, [RULE]) is None


def test_independent_reciprocal_action_remains_verifiable() -> None:
    claim = "Региональный совет направляет заключение."
    for reciprocal in [
        "Региональный совет направляет заключение комитету.",
        "Региональный совет:\n(i) направляет заключение комитету.",
        "Заключение направляется комитету региональным советом.",
    ]:
        assert reversed_list_actor(claim, [RULE, reciprocal]) is None
