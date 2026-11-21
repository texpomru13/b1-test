"""Guard explicit Russian object counts against quotes measuring another quantity."""

from __future__ import annotations

import re

from py_rust_stemmers import SnowballStemmer

_STEMMER = SnowballStemmer("russian")
_WORD = r"[а-яё]+"
_ASKED_COUNT = re.compile(rf"^\s*сколько\s+({_WORD})\b", re.I)
_NUMBER_WORDS = (
    "ноль нуля", "один одна одно одну одного одной", "два две двух", "три трех",
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
_FREQUENCY = re.compile(r"\b(?:однократно|единожды|единовременно|однажды)\b")
_PER = re.compile(r"\b(?:на|за)\s+$")
_ENTITLEMENT = re.compile(r"\b(?:право|разрешение|претендовать|рассчитывать)\s+на\s+$")
_PER_EACH = re.compile(rf"\b(?:на|за)\s+кажд{_WORD}\s+(?:отдельн{_WORD}\s+)?({_WORD})")
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


def _other_measure(quote: str, number: str, unit: set[str]) -> bool:
    """True only for a recognizable competing measure without a supporting item count."""
    quote = quote.lower().replace("ё", "е")
    denominator = any(
        not _unit_forms(m[1]).isdisjoint(unit) for m in _PER_EACH.finditer(quote)
    )
    competing = bool(_FREQUENCY.search(quote))
    other_item = False
    counts = list(_COUNT.finditer(quote))
    for match in counts:
        value = _NUMBERS.get(match["value"], match["value"].replace(",", "."))
        measured = _unit_forms(match["unit"])
        prefix = quote[:match.start()]
        per_item = _PER.search(prefix) and not _ENTITLEMENT.search(prefix)
        if not measured.isdisjoint(_MEASURES):
            competing = True
        elif not measured.isdisjoint(unit) and per_item:
            # "на одну камеру предоставляется период" gives the denominator.
            denominator |= len(counts) > 1 or competing
        elif value == number and not per_item:
            if not measured.isdisjoint(unit):
                return False
            other_item = True  # possible synonym or legal designation
    return denominator or (competing and not other_item)


def unsupported_object_count(question: str, claim: str, quotes: list[str]) -> bool:
    """Is the claimed item count supported only by quotes measuring something else?"""
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
        if all(_other_measure(quote, number, unit) for quote in quotes):
            return True
    return False
