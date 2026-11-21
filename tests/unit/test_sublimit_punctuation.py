import pytest
from app.qa.grounding import LocatedQuote
from app.qa.limits import restore_sublimit_punctuation
from app.retrieval.search import ContextChunk


def quote(text, end_page=1):
    c = ContextChunk("p", 0, text, None, 1, end_page, 120, 120 + len(text), .9)
    return LocatedQuote(c, text, 120, 120 + len(text), 1, end_page)


@pytest.mark.parametrize(("overall", "per_job"), [
    ("8% от бюджета склада", "3% от размера заказа"),
    ("3% от размера заказа", "8% от бюджета склада"),
    ("7% от годовой суммы", "7% от разового обеспечения"),
])
def test_two_bases_stay_on_their_own_sides_and_exception_stays_outside(overall, per_job):
    norm = (f"Компенсация ограничена {overall} (для одной партии — в пределах {per_job}), "
            "если соглашением не предусмотрено иное.")
    draft = norm.replace("партии —", "партии) –").replace("), если", ", если")
    actual = restore_sublimit_punctuation(draft, [quote(norm)])
    assert actual == norm
    assert actual.count("%") == 2


def test_running_title_and_folio_can_split_one_parenthesis_in_a_citation():
    norm = ("Оплата ограничена 12% от бюджета фонда (на один рейс — до 4% от стоимости рейса), "
            "если не согласовано иное.")
    raw = norm.replace("рейс —", "рейс\nПравила перевозки грузов\n\n 2\n—")
    draft = norm.replace("рейс —", "рейс) –").replace("рейса),", "рейса,")
    assert restore_sublimit_punctuation(draft, [quote(raw, 2)]) == norm
    changed = draft.replace("от бюджета фонда", "от стоимости рейса")
    with pytest.raises(ValueError, match="база"):
        restore_sublimit_punctuation(changed, [quote(raw, 2)])


def test_an_independent_parenthesis_or_an_extra_real_condition_is_not_relocated():
    draft = "Взнос составляет 6% от оборота (на одну сделку) — до 2% от резерва."
    unrelated = "Взнос не превышает 6% от оборота (по другому соглашению — до 3% от резерва)."
    assert restore_sublimit_punctuation(draft, [quote(unrelated)]) == draft
    source = "Взнос составляет 6% от оборота (на одну сделку при согласовании — до 2% от резерва)."
    assert restore_sublimit_punctuation(draft, [quote(source)]) == draft
