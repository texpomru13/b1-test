from app.qa.template_fields import clarify_unfilled_duration

NOTICE = "Поставщик подаёт заявку не менее чем за ____ до даты отгрузки."
CLAIM = "Поставщик обязан подать заявку за определённый срок до отгрузки."


def test_blank_duration_is_visible_alongside_the_substantive_rule() -> None:
    text = clarify_unfilled_duration(CLAIM, [NOTICE])
    assert NOTICE in text
    assert "Поле для указания срока в этом пункте не заполнено" in text
    assert "определённый срок" not in text


def test_another_cited_clause_can_supply_the_duration() -> None:
    assert clarify_unfilled_duration(CLAIM, [
        NOTICE, "Срок подачи заявки по приложению составляет 10 рабочих дней."
    ]) == CLAIM
    assert clarify_unfilled_duration(CLAIM, ["Заявка подаётся за 5 дней."]) == CLAIM


def test_unrelated_blanks_and_claims_stay_unchanged() -> None:
    assert clarify_unfilled_duration(CLAIM, ["Менеджер ____ принял заявку."]) == CLAIM
    plain = "Заявку подаёт поставщик."
    assert clarify_unfilled_duration(plain, [NOTICE]) == plain
