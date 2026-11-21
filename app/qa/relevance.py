"""Bind requested limits and supplementary facts to the cited subject and rule."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from py_rust_stemmers import SnowballStemmer

from app.qa.glosses import without_unproven_gloss
from app.qa.grounding import LocatedQuote, ProvenanceError, quote_at_offsets
from app.qa.limits import restore_sublimit_punctuation
from app.qa.schemas import Claim
from app.retrieval.search import ContextChunk

_STEMMER = SnowballStemmer("russian")
_WORD = re.compile(r"[а-яё]{3,}|[a-z]{3,}", re.I)
_STOP = frozenset(("какой какая какие какого каким каких какова каковы каков сколько ли "
                   "или для при это этой этим иные уже при этом она него "
                   "может можно нужно должны вправе согласно случай условиях условий договор "
                   "договору договора заказчик стороны сторонам общий общие новых новую ранее "
                   "указаны указано данный этого норма пункта пункт статье статьи").split())
_UPPER = r"максим(?:альн\w*|ум)|наибольш\w*|верхн\w*\s+(?:границ\w*|предел\w*)"
_LOWER = r"миним(?:альн\w*|ум)|наименьш\w*|нижн\w*\s+(?:границ\w*|предел\w*)"
_REQUEST = re.compile(rf"\b(?:(?P<upper>{_UPPER})|(?P<lower>{_LOWER}))", re.I)
_BOUND = re.compile(r"\b(?:(?P<upper>не\s+(?:более|больше|выше)|до|максимум)"
                    r"|(?P<lower>не\s+(?:менее|меньше|ниже)|от|минимум))\s*$", re.I)
_UNITS = {
    "kg": r"кг|килограмм\w*", "t": r"тн|т|тонн\w*", "kgf": r"кгс",
    "kw": r"квт|киловатт\w*", "w": r"вт|ватт\w*", "v": r"в|вольт\w*",
    "m": r"м|метр\w*", "km": r"км|километр\w*", "l": r"л|литр\w*",
    "percent": r"%|процент\w*", "money": r"руб\.?|рубл\w*|₽",
    "day": r"(?:(?:календарн|рабоч)\w*\s+)?(?:день|дня|дней|дни)",
    "hour": r"час\w*", "minute": r"мин\.|минут\w*", "month": r"месяц\w*",
}
_UNIT = "|".join(f"(?P<{k}>{v})" for k, v in _UNITS.items())
_MEASURE = re.compile(r"(?<![\w.,])(?P<number>\d+(?:[.,]\d+)?)(?:\s*\([^\d()]{1,45}\))?"
                      rf"\s*(?:{_UNIT})(?!\w)", re.I)
_UNIT_ONLY = re.compile(rf"(?<!\w)(?:{_UNIT})(?!\w)", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁ](?:[а-яё]|\s+[а-яё]))")
_SOURCE = (r"(?P<appendix>приложени[еяиюй]\s*(?:№\s*)?(?P<number>\d+(?:\.\d+)*)(?!\d|\.\d))"
           r"|(?P<main>основн\w*\s+(?:текст\w*\s+)?договор\w*)")
_SOURCE_PRIORITY = (
    re.compile(rf"(?:{_SOURCE})\s+(?:име\w+\s+приоритет\b|"
               r"име\w+\s+(?:преимущественн\w*|приоритетн\w*)\s+сил\w+|"
               r"преоблада\w+\s+над\b)", re.I),
    re.compile(r"\b(?:при|в случае)\s+(?:расхожден\w+|противоречи\w+|несоответстви\w+)"
               r"[^.!?;]{0,210}?\b(?:применя\w+|руководству\w+)\s+"
               rf"(?:(?:услови\w*|положени\w*|текст\w*)\s+)?(?:{_SOURCE})", re.I),
)
_ASK_BOTH = re.compile(r"\b(?:сравни\w*|сопостав\w*|чем отлича\w*|в чем разниц\w*)\b", re.I)
_ASK_MAIN = re.compile(r"\b(?:по(?: тексту)?|что указано в)\s+основн\w*\s+договор\w*", re.I)
_QUALIFIES_CLAUSE = re.compile(
    r"\b(?:если|только|исключительно|кроме|за исключением|при|без|для|когда|"
    r"в случае|в зависимости от|до тех пор)\b", re.I,
)
_COMPARE = re.compile(r"\b(?:сравн\w*|сопостав\w*|различ\w*|расхожден\w*|"
                      r"противореч\w*|разниц\w*|отлич\w*|форм[аеуы]|образц\w*|"
                      r"приложени\w*|акт[аеу]?)\b", re.I)
_CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:[-–—]|до)\s*\d{1,2}[:.]\d{2}\b")
_OWNER = re.compile(r"^\s*(?:\d+[.)]\s+)?(?P<owner>[А-ЯЁA-Z][\w ()«»]{2,65}?)"
                    r"\s*(?:[–—]|\s-\s|:\s)")
_ANALOGY = re.compile(r"\b(?:аналогичн\w*|по аналогии|так(?:ой|ая|ое|ие) же)\b", re.I)
_TIME = re.compile(r"\b(?:время|времени|срок\w*|дн\w*|сут\w*|час\w*|минут\w*)\b", re.I)
_TRIGGER_REF = re.compile(r"\b(?:при|если|в случае)\s+[^.;:]{3,140}?"
                          r"\b(?:п\.|пунктах?\b|пункте\b|статье\b)\s*(\d+(?:\.\d+)*)", re.I)

# Compare what is being changed, not just the shared noun or procedure.
# Synonyms of one attribute remain eligible; an unspecified attribute does not
# create a conflict. The object itself must occur in both noun phrases.
_PROPERTIES = {
    "term": r"срок\w*|продолжительност\w*|длительност\w*|период\w*",
    "price": r"стоимост\w*|цен[а-яё]*|тариф\w*",
    "quantity": r"количеств\w*|объ[её]м\w*|числ[а-яё]*",
    "place": r"адрес\w*|мест[а-яё]*|местоположени\w*",
}
_CHANGE = re.compile(r"\b(?:изменени|пересмотр|корректировк|продлени|согласовани)"
                     r"\w*\s+(?P<attribute>" + "|".join(_PROPERTIES.values())
                     + r")\s+(?P<object>[а-яё][а-яё-]{2,})\b", re.I)


@lru_cache(maxsize=8192)
def _stem(word: str) -> str:
    return _STEMMER.stem_word(word.casefold().replace("ё", "е"))


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text) if w.casefold() not in _STOP]


def _measures(text: str) -> set[tuple[str, str]]:
    return {(m["number"].replace(",", "."), str(m.lastgroup)) for m in _MEASURE.finditer(text)}


def _requested_bound(
    question: str, claims: list[Claim], located: dict[int, list[LocatedQuote]],
) -> None:
    # A minimum provided as a premise does not make this a request for a minimum.
    lead = re.split(r"\b(?:если|когда|при условии)\b", question, maxsplit=1, flags=re.I)[0]
    directions = {m.lastgroup for m in _REQUEST.finditer(lead)}
    if len(directions) != 1:
        return
    wanted = directions.pop()
    units = {m.lastgroup for m in _UNIT_ONLY.finditer(lead)
             if len(m[0]) > 1 or m[0] in {"%", "₽"}}
    evidence_sides: set[str] = set()
    for i, claim in enumerate(claims):
        stated = _measures(claim.text)
        for quote in located[i]:
            for value in _MEASURE.finditer(quote.quote_raw):
                fact = (value["number"].replace(",", "."), str(value.lastgroup))
                if fact not in stated or (units and value.lastgroup not in units):
                    continue
                prefix = quote.quote_raw[:value.start()]
                bound = _BOUND.search(prefix)
                if bound:
                    evidence_sides.add(str(bound.lastgroup))
                    continue
                # A noun-bound limit may precede the value and its linking verb.
                tail = re.split(r"[.,;:\n!?\d]", prefix)[-1][-80:]
                adjectives = list(_REQUEST.finditer(tail))
                if len(adjectives) == 1:
                    evidence_sides.add(str(adjectives[0].lastgroup))
    if evidence_sides and wanted not in evidence_sides:
        side = "верхний" if wanted == "upper" else "нижний"
        raise ProvenanceError(f"Запрошен {side} предел. Для названной величины в цитате "
                              "подтверждена только противоположная граница. Выбери "
                              "применимый предел или ответь answerable=false.")


def explicit_source_priority(sources: dict[str, ContextChunk]) -> str | None:
    """Return the preferred appendix number, an empty string for the main text."""
    priority = {
        (match["number"] or "")
        for chunk in sources.values()
        for pattern in _SOURCE_PRIORITY
        for match in pattern.finditer(" ".join(chunk.text.split()))
    }
    return priority.pop() if len(priority) == 1 else None


def _source_number(chunk: ContextChunk) -> str | None:
    if not chunk.scope_id and not chunk.scope_label:
        return ""
    match = re.match(r"^приложение\s*(?:№\s*)?(\d+(?:\.\d+)*)\b",
                     chunk.scope_label or "", re.I)
    return match[1] if match else None


def _priority_measurements(
    sources: dict[str, ContextChunk], number: str,
) -> list[tuple[set[str], set[tuple[str, str]]]]:
    return [
        (set(_terms(sentence)), _measures(sentence))
        for chunk in sources.values() if _source_number(chunk) == number
        for sentence in _SENTENCE.split(chunk.text) if _MEASURE.search(sentence)
    ]


def narrow_priority_evidence(
    question: str, claims: list[Claim], located: dict[int, list[LocatedQuote]],
    sources: dict[str, ContextChunk], number: str,
) -> None:
    """Cite an exact independent clause without an adjacent conflicting measurement."""
    if _ASK_BOTH.search(question) or _ASK_MAIN.search(question):
        return
    authority = _priority_measurements(sources, number)
    if not authority:
        return
    question_terms = set(_terms(question))
    for index, claim in enumerate(claims):
        if _measures(claim.text):
            continue
        words = claim.text.rstrip(".!? ").split()
        if len(words) < 2:
            continue
        exact = re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"[.!?]?\s*$", re.I)
        for pos, quote in enumerate(located[index]):
            match = exact.search(quote.quote_raw)
            if _source_number(quote.chunk) == number or match is None:
                continue
            prefix = _SENTENCE.split(quote.quote_raw[:match.start()])[-1].rstrip()
            if not prefix.endswith((",", ";")) or _QUALIFIES_CLAUSE.search(prefix):
                continue
            values = _measures(prefix)
            topic = question_terms & set(_terms(prefix))
            if not any(len(topic & terms) >= 2
                       and any(a != b and unit_a == unit_b
                               for a, unit_a in values for b, unit_b in measures)
                       for terms, measures in authority):
                continue
            offset = quote.char_start - quote.chunk.char_start
            end = len(quote.quote_raw.rstrip())
            located[index][pos] = quote_at_offsets(quote.chunk, offset + match.start(),
                                                   offset + end)


def _validate_source_priority(
    question: str, claims: list[Claim], located: dict[int, list[LocatedQuote]],
    sources: dict[str, ContextChunk],
) -> None:
    """Reject a conflicting fact when the selected context establishes precedence."""
    if _ASK_BOTH.search(question) or _ASK_MAIN.search(question):
        return
    number = explicit_source_priority(sources)
    if number is None:
        return

    authority = _priority_measurements(sources, number)
    for index, claim in enumerate(claims):
        measured = _measures(claim.text)
        if len(measured) != 1:
            continue
        value, dimension = next(iter(measured))
        subject = (set(_terms(claim.text)) & set(_terms(question))
                   & set(_terms(" ".join(q.quote_raw for q in located[index]))))
        alternatives = {
            amount for terms, facts in authority if len(subject.intersection(terms)) >= 2
            for amount, unit in facts if unit == dimension
        }
        if len(alternatives) == 1 and value not in alternatives:
            raise ProvenanceError(
                "Документ явно устанавливает приоритет источника. Приведи значение "
                "из применимого приоритетного пункта и сошлись на него."
            )


def _different_attribute(question: str, local: str, direct: str) -> bool:
    if _COMPARE.search(question):
        return False
    direct_terms = set(_terms(direct))
    for asked in _CHANGE.finditer(question):
        target = _stem(asked["object"])
        if target not in direct_terms or _stem(asked["attribute"]) not in direct_terms:
            continue
        requested_family = next(pattern for pattern in _PROPERTIES.values()
                                if re.fullmatch(pattern, asked["attribute"], re.I))
        if re.search(r"\b(?:" + requested_family + r")\b", local, re.I):
            continue
        for rule in _CHANGE.finditer(local):
            if _stem(rule["object"]) != target:
                continue
            for property_words in _PROPERTIES.values():
                if re.fullmatch(property_words, rule["attribute"], re.I):
                    if (not re.fullmatch(property_words, asked["attribute"], re.I)
                            and not re.search(r"\b(?:" + property_words + r")\b", question, re.I)):
                        return True
    return False


def _redundant_branch(
    question: str, statement: str, quotes: list[LocatedQuote], direct: list[LocatedQuote],
) -> bool:
    other_text = " ".join(q.quote_raw for q in direct)
    qterms = Counter(_terms(question))
    other_terms = set(_terms(other_text))
    if len(qterms.keys() & other_terms) < 2:
        return False
    local_text = " ".join(q.quote_raw for q in quotes)
    if (_ANALOGY.search(statement) and not _ANALOGY.search(local_text)
            and not _COMPARE.search(question) and not _ANALOGY.search(question)):
        return True
    if _different_attribute(question, local_text, other_text):
        return True
    # A labeled schedule establishes times for the named service/resource only.
    # Keep the row if that owner is explicitly requested or connected by a direct rule.
    for line in local_text.splitlines():
        if not _CLOCK.search(line) or not (owner := _OWNER.match(line)):
            continue
        names = set(_terms(owner["owner"]))
        # Inflection of a short label can collide with a verbal suffix in a stemmer.
        explicit = any(re.search(r"\b" + re.escape(w) + r"\w*\b", question + " " + other_text,
                                 re.I) for w in _WORD.findall(owner["owner"]) if len(w) >= 4)
        if (names and not explicit and names.isdisjoint(qterms) and names.isdisjoint(other_terms)
                and _TIME.search(other_text) and not _CLOCK.search(other_text)):
            return True
    # A branch explicitly conditioned on circumstances from a different numbered
    # clause cannot define the repeated subject of the question by analogy.
    topic = {term for term, count in qterms.items() if count > 1 and term in other_terms}
    trigger = _TRIGGER_REF.search(local_text)
    return bool(trigger and topic and topic.isdisjoint(_terms(local_text))
                and not re.search(r"(?<![\d.])" + re.escape(trigger[1]) + r"(?![\d.])",
                                  question + " " + other_text))


def refine_answer_scope(question: str, claims: list[Claim], located: dict[int, list[LocatedQuote]],
                        sources: dict[str, ContextChunk]) -> dict[int, list[LocatedQuote]]:
    """Keep a direct answer with its own context; the semantic verifier still checks coverage."""
    for index, claim in enumerate(claims):
        try:
            claim.text = without_unproven_gloss(
                claim.text, [q.quote_raw for q in located[index]], question,
            )
            claim.text = restore_sublimit_punctuation(claim.text, located[index])
        except ValueError as exc:
            raise ProvenanceError(str(exc)) from exc
    _requested_bound(question, claims, located)
    _validate_source_priority(question, claims, located, sources)
    if len(claims) < 2:
        return located
    keep = []
    for i in range(len(claims)):
        local_ids = {q.chunk.chunk_id for q in located[i]}
        other = [q for j, quotes in located.items() if i != j for q in quotes
                 if q.chunk.chunk_id not in local_ids]
        if not other or not _redundant_branch(question, claims[i].text, located[i], other):
            keep.append(i)
    if not keep:
        return located
    claims[:] = [claims[i] for i in keep]
    return {idx: located[i] for idx, i in enumerate(keep)}
