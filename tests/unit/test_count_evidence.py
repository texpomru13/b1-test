import pytest
from app.qa.counts import unsupported_object_count


@pytest.mark.parametrize(("question", "claim", "quote"), [
    ("Сколько пропусков выдаётся?", "Выдаётся один пропуск", "Подать заявление можно однократно."),
    ("Сколько билетов доступно?", "Доступны 2 билета", "Участвовать можно два раза."),
    ("Сколько терминалов выдаётся?", "Один терминал",
     "На один терминал предоставляется один пропуск."),
    ("Сколько коек доступно?", "Доступны три койки", "Обращение рассматривается 3 дня."),
])
def test_frequency_time_or_denominator_cannot_support_an_object_count(
    question: str, claim: str, quote: str
) -> None:
    assert unsupported_object_count(question, claim, [quote])


@pytest.mark.parametrize(("question", "claim", "quotes"), [
    ("Сколько обращений разрешено?", "Одно обращение", ["Пользователь может подать 1 обращение."]),
    ("Сколько билетов положено?", "Один билет", ["На одно обращение выдаётся один билет."]),
    ("Сколько датчиков предоставляется?", "Три датчика", [
        "В программе участвуют один раз.",
        "Допускается получение не более 3 (трёх) единиц Оборудования.",
    ]),
    ("Сколько приборов имеется?", "Один прибор", ["Комплект включает одно устройство."]),
    ("Сколько билетов выдаётся?", "Один билет", ["Участник имеет право на один билет."]),
    ("Сколько раз можно обратиться?", "Один раз", ["В рамках акции обращаются однократно."]),
    ("Сколько времени дано на оплату?", "Три дня", ["Заявку можно оформить один раз."]),
    ("Сколько рублей стоит билет?", "Три рубля", ["Выдача производится один раз."]),
])
def test_supported_ambiguous_or_different_measure_is_left_to_the_verifier(
    question: str, claim: str, quotes: list[str]
) -> None:
    assert not unsupported_object_count(question, claim, quotes)


def test_per_unit_and_neutral_citations_cannot_establish_the_quantity_allocated() -> None:
    question = "Сколько жетонов может получить посетитель?"
    claim = "Посетитель может получить только один жетон."
    quotes = ["Зарегистрироваться можно единовременно.", "Повторная регистрация запрещена.",
              "Оплата обслуживания производится в расчете на один жетон."]
    assert unsupported_object_count(question, claim, quotes)
    assert unsupported_object_count(question, claim, ["Выдача осуществляется по заявке."])
    assert not unsupported_object_count(question, claim, [*quotes, "Положен один жетон."])
    assert not unsupported_object_count(question, claim, [
        *quotes, "Посетитель имеет право на один электронный пропуск."])
