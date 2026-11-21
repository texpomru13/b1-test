import pytest
from app.qa.assemble import assemble_answer
from app.qa.numbers import qualify_numeric_text
from app.qa.workflow import run_qa

from tests.unit.test_qa_workflow import SCOPE, ScriptedAdapter, make_deps

PRICE = ("Таблица № 2. Стоимость датчиков с единовременной оплатой\nдля всех каналов продаж\n"
         "Оборудование Стоимость, руб. с НДС\nДатчик 7 250")
DURATION = ("Максимальная продолжительность пробного периода составляет 12 (двенадцать) "
            "календарных дней с даты подключения.")


async def test_price_term_and_upper_bound_reach_both_verifier_and_answer() -> None:
    adapter = ScriptedAdapter([
        {"answerable": True, "missing_information": [], "claims": [
            {"text": "Датчик стоит 7 250 руб. с НДС.", "evidence": ["source-1.1"]},
            {"text": "Пробный период составляет 12 календарных дней с даты подключения.",
             "evidence": ["source-2.1"]}]},
        {"coverage": "full", "verdicts": [
            {"claim_index": 0, "verdict": "supported", "reason": ""},
            {"claim_index": 1, "verdict": "supported", "reason": ""}]},
    ])
    state = await run_qa(make_deps(adapter, [PRICE, DURATION]),
                         question="Сколько стоит датчик и каков пробный период?",
                         scope=SCOPE, debug=False)
    response = assemble_answer(state)
    assert state.llm_calls == 2
    assert response.citations[0].text == PRICE and response.citations[1].text == DURATION
    for qualifier in ("при единовременной оплате", "не более 12 календарных дней"):
        assert qualifier in response.answer and qualifier in adapter.prompts[1]


def test_bounds_and_payment_apply_to_the_matching_value_and_role() -> None:
    exact = "Срок составляет 6 дней."
    assert qualify_numeric_text(exact, ["Максимальная цена 6 рублей.", exact]) == exact
    restored = qualify_numeric_text("Срок 6 дней.", ["Минимальный срок 6 дней."])
    assert restored == "Срок не менее 6 дней."
    assert qualify_numeric_text("Не более 12 дней.", [DURATION]) == "Не более 12 дней."
    with pytest.raises(ValueError, match="предел"):
        qualify_numeric_text("Минимальный срок 12 дней.", [DURATION])
    with pytest.raises(ValueError, match="Раздели"):
        qualify_numeric_text(exact, ["Минимальный срок 6 дней.", exact])
    with pytest.raises(ValueError, match="оплаты"):
        qualify_numeric_text("В рассрочку датчик стоит 7250 рублей.", [PRICE])
    other = ("Таблица № 3. Стоимость кабеля\nОборудование Стоимость, руб. с НДС\nКабель 10")
    assert qualify_numeric_text("Кабель стоит 10 рублей.", [PRICE + "\n" + other]) == (
        "Кабель стоит 10 рублей."
    )
    with pytest.raises(ValueError, match="оплаты"):
        same_price = other.replace("Кабель 10", "Кабель 7250")
        qualify_numeric_text("Цена 7250 рублей.", [PRICE, same_price])


@pytest.mark.parametrize("reverse", [False, True])
def test_rate_applies_to_its_own_interval_even_when_bounds_repeat(reverse: bool) -> None:
    rows = ["на срок от 91 до 180 дней — 15,15%",
            "на срок от 181 дня до 1 года — 15,85%",
            "на срок свыше 1 года — 15,05%"]
    quote = "; ".join(reversed(rows) if reverse else rows)
    valid = "Для срока свыше одного года ставка составляет 15,05%."
    assert qualify_numeric_text(valid, [quote]) == valid
    both = "Для срока от 91 до 180 дней — 15,15%, для срока свыше 1 года — 15,05%."
    assert qualify_numeric_text(both, [quote]) == both
    for invalid in ("Для срока свыше 1 года ставка 15,85%.",
                    "Для срока до 1 года ставка 15,05%.",
                    "Ставка составляет 15,05%.",
                    "Для сроков от 91 до 180 дней и свыше 1 года ставка 15,05%."):
        with pytest.raises(ValueError, match=r"[Дд]иапазон"):
            qualify_numeric_text(invalid, [quote])


def test_band_in_other_quote_cannot_supply_unrelated_or_ambiguous_table_rate() -> None:
    row = "От 10 до 12 месяцев: 3,7%"
    with pytest.raises(ValueError, match="одной строкой"):
        qualify_numeric_text("Свыше 12 месяцев — 3,7%.",
                            [row, "Свыше 12 месяцев: 4,2%"])
    with pytest.raises(ValueError, match="одной строкой"):
        qualify_numeric_text("От 10 до 12 месяцев — 3,7%.",
                            [row, "От 10 до 12 месяцев: 4,2%"])
