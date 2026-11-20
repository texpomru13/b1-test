import pytest
from app.qa.counts import enforce_count_evidence
from app.schemas.qa import AnswerResponse, Citation


def answer(claim: str, *quotes: str) -> AnswerResponse:
    citations = [Citation(
        id=i, text=text, page=1, page_to=1, section=None, chunk_id=f"c{i}",
        char_start=0, char_end=len(text),
    ) for i, text in enumerate(quotes, 1)]
    markers = "".join(f"[{c.id}]" for c in citations)
    return AnswerResponse(answer=f"{claim}. {markers}", found=True, citations=citations)


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
    filtered = enforce_count_evidence(question, answer(claim, quote))
    assert filtered.found is False and not filtered.citations


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
    original = answer(claim, *quotes)
    assert enforce_count_evidence(question, original) is original
