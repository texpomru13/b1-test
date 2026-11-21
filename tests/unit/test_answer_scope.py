import pytest
from app.qa.grounding import ProvenanceError, locate_quote
from app.qa.relevance import refine_answer_scope
from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk


def assess(question, rows):
    sources = {}
    claims = []
    refs = {}
    for i, (statement, quote, scope) in enumerate(rows):
        key = f"source-{i + 1}"
        chunk = ContextChunk(key, i, quote, None, 1, 1, 300 * i, 300 * i + len(quote), 1.0,
                             scope_id=scope, scope_label=scope)
        sources[key] = chunk
        claims.append(Claim(text=statement, evidence=[f"{key}.1"]))
        refs[i] = [locate_quote(chunk, quote)]
    out = refine_answer_scope(question, claims, refs, sources)
    assert list(out) == list(range(len(claims)))
    return [c.text for c in claims], out


@pytest.mark.parametrize(("direction", "wrong", "correct"), [
    ("максимальный", "не менее", "не более"), ("минимальный", "не больше", "не меньше"),
])
def test_opposite_limit_does_not_answer_even_if_claim_is_verbatim(direction, wrong, correct):
    question = f"Какой {direction} вес посылки в килограммах?"
    bad = f"Вес посылки {wrong} 17 кг."
    with pytest.raises(ProvenanceError, match="граница"):
        assess(question, [(bad, bad, None)])
    valid = f"Вес посылки {correct} 25 кг."
    statements, _ = assess(question, [(valid, valid + " " + bad, None)])
    assert statements == [valid]


@pytest.mark.parametrize("reverse", [True, False])
def test_appendix_with_explicit_priority_preserves_18_liters_and_its_quote(reverse):
    q = "Какова вместимость резервной цистерны и что входит в обслуживание цистерны?"
    alt = ("Вместимость резервной цистерны составляет 18 л.",
           "2. Вместимость резервной цистерны 18 л.", "Приложение № 8")
    primary = ("В обслуживание входит очистка емкости.",
               "Работы согласованы. Резервная цистерна вмещает 40 л, в обслуживание входит "
               "очистка цистерны. При расхождении применяются условия Приложения № 8.", None)
    rows = [alt, primary] if reverse else [primary, alt]
    statements, refs = assess(q, rows)
    i = rows.index(alt)
    assert statements[i] == alt[0]
    quote = refs[i][0]
    assert quote.quote_raw == alt[1]
    assert quote.chunk.scope_label == "Приложение № 8"
    wrong = ("Резервная цистерна вмещает 40 л.", primary[1], None)
    with pytest.raises(ProvenanceError, match="явно устанавливает приоритет"):
        assess(q, [wrong, alt])
    compared, _ = assess("Сравните приложение и договор. " + q, [wrong, alt])
    assert compared == [wrong[0], alt[0]]
    neutral = ("Резервная цистерна вмещает 40 л.",
               primary[1].split("При расхождении")[0], None)
    assert assess(q, [neutral, alt])[0] == [neutral[0], alt[0]]


@pytest.mark.parametrize("reverse", [True, False])
def test_owned_timetable_is_not_a_general_deadline(reverse):
    q = "В каком режиме и в какой срок возвращают контейнер?"
    specific = ("Контейнер возвращают в рабочее время в течение суток.",
                "После проверки контейнер возвращают в рабочее время в течение суток.", None)
    table = ("Часы работы с 07:30 до 12:45.",
             "Телефоны:\nАрхив — с 07:30 до 12:45.", None)
    rows = [specific, table] if reverse else [table, specific]
    assert assess(q, rows)[0] == [specific[0]]
    assert len(assess(q + " Назовите также часы работы архива.", rows)[0]) == 2


def test_cross_reference_to_a_distinct_event_cannot_extend_requested_rule():
    q = "Реестр изменён. Какая редакция действует и как выбрать редакцию?"
    relevant = ("Остаётся согласованная редакция реестра.",
                "Остаётся согласованная редакция реестра, стороны могут выбрать иную.", None)
    foreign = ("Оператор письменно извещает стороны.",
               "При наступлении обстоятельств из пункта 8.2 оператор письменно извещает стороны.",
               None)
    assert assess(q, [foreign, relevant])[0] == [relevant[0]]
    assert len(assess(q + " Как уведомить об обстоятельствах из пункта 8.2?",
                      [foreign, relevant])[0]) == 2


def test_unrequested_analogy_cannot_extend_a_specific_procedure():
    direct = ("Для смены места обучения требуется согласование.",
              "Смена места обучения оформляется письменным согласованием.", None)
    other = ("При переводе кредита оформляется заявка. Аналогичная процедура действует при учебе.",
             "При переводе кредита оформляется заявка.", None)
    question = "Как оформить смену места обучения?"
    assert assess(question, [direct, other])[0] == [direct[0]]
    assert len(assess(question + " Сравните с переводом кредита.", [direct, other])[0]) == 2


def test_minimum_for_another_unit_does_not_contradict_requested_maximum():
    question = "Какова максимальная высота опоры в метрах?"
    low = ("Масса не менее 12 кг.", "Масса не менее 12 кг.", None)
    upper = ("Максимальная высота — 4 м.", "Максимальная высота — 4 м.", None)
    assert len(assess(question, [low, upper])[0]) == 2
