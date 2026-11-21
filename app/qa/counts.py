"""Guard explicit Russian object counts against quotes measuring another quantity."""

from __future__ import annotations

import re

from py_rust_stemmers import SnowballStemmer

_STEMMER = SnowballStemmer("russian")
_WORD = r"[а-яё]+"
_ASKED_COUNT = re.compile(rf"^\s*сколько\s+({_WORD})\b", re.I)
_NUMBER_WORDS = (
    "ноль нуля", "один одна одно одну одного одной одному одним", "два две двух", "три трех",
    "четыре четырех", "пять пяти", "шесть шести", "семь семи", "восемь восьми",
    "девять девяти", "десять десяти", "одиннадцать", "двенадцать", "тринадцать",
    "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать",
    "девятнадцать", "двадцать",
)
_NUMBERS = {w: str(n) for n, forms in enumerate(_NUMBER_WORDS) for w in forms.split()}
_NUMBER = r"\d+(?:[.,]\d+)?|" + "|".join(_NUMBERS)
_COUNT = re.compile(
    rf"\b(?P<value>{_NUMBER})\b(?:\s*\([^()\n]{{1,60}}\))?\s+"
    rf"(?:(?:отдельн{_WORD}|дополнительн{_WORD})\s+)?(?P<unit>{_WORD}|[%₽€$])"
)

_PER = re.compile(r"\b(?:на|за)\s+$")
_ENTITLEMENT = re.compile(r"\b(?:право|разрешение|претендовать|рассчитывать)\s+на\s+$")
_DENOMINATOR = re.compile(r"\b(?:в\s+(?:пере)?расчете|из\s+расчета)\s+(?:на|за)\s+$")
_ALLOCATED = re.compile(r"\b(?:предостав|получ|выда|выдан|полага|положен|переда|включа|"
                        r"доступ|допуска|разреш|право|содерж|состо|рассчитыва|приобрета)\w*")
_GENERIC_ITEMS = {"единиц", "штук", "экземпляр"}
# These units of time, frequency, money and measurement cannot establish an object count.
_MEASURES = {
    _STEMMER.stem_word(x) for x in
    "раз раза разов лет год день дня дни дней сутки суток месяц неделя час минута секунда "
    "рубль рублей тенге доллар евро процент процента время времени срок".split()
} | {"%", "₽", "€", "$"}
_NON_OBJECT = _MEASURES | {
    _STEMMER.stem_word(x) for x in "стоит стоят рабочих календарных денег".split()
}


def _unit_forms(word: str) -> set[str]:
    forms = {word, _STEMMER.stem_word(word)}
    if word.endswith(("ок", "ек")):  # заявок/заявка, коек/койка
        forms.update((word[:-2] + "к", word[:-2] + "йк"))
    return forms


def _supports_count(quote: str, number: str, unit: set[str]) -> bool:
    """Require a matching numerator, allowing object synonyms in allocation clauses."""
    quote = quote.casefold().replace("ё", "е")
    counts = list(_COUNT.finditer(quote))
    denominator_named = any(
        _PER.search(quote[:m.start()]) and not _ENTITLEMENT.search(quote[:m.start()])
        and not _unit_forms(m["unit"]).isdisjoint(unit) for m in counts
    )
    for match in counts:
        value = _NUMBERS.get(match["value"], match["value"].replace(",", "."))
        if value != number:
            continue
        measured = _unit_forms(match["unit"])
        prefix = quote[:match.start()]
        if _DENOMINATOR.search(prefix) or (_PER.search(prefix) and not _ENTITLEMENT.search(prefix)):
            continue
        if not measured.isdisjoint(_NON_OBJECT):
            continue
        if not measured.isdisjoint(unit):
            return True
        if denominator_named:
            continue
        clause_start = max(prefix.rfind("."), prefix.rfind(";"), prefix.rfind("!")) + 1
        if not measured.isdisjoint(_GENERIC_ITEMS) or _ALLOCATED.search(prefix[clause_start:]):
            return True
    return False


def unsupported_object_count(question: str, claim: str, quotes: list[str]) -> bool:
    """Does an explicit count of the requested object lack a supporting quantity?"""
    asked = _ASKED_COUNT.match(question.lower().replace("ё", "е"))
    if asked is None:
        return False
    unit = _unit_forms(asked[1])
    if not unit.isdisjoint(_NON_OBJECT):
        return False
    for match in _COUNT.finditer(claim.lower().replace("ё", "е")):
        if _unit_forms(match["unit"]).isdisjoint(unit):
            continue
        number = _NUMBERS.get(match["value"], match["value"].replace(",", "."))
        if not any(_supports_count(quote, number, unit) for quote in quotes):
            return True
    return False
