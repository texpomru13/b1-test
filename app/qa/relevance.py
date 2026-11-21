"""Bind requested limits and supplementary facts to the cited subject and rule."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache

from py_rust_stemmers import SnowballStemmer

from app.qa.evidence import issue_evidence
from app.qa.glosses import without_unproven_gloss
from app.qa.grounding import LocatedQuote, ProvenanceError, quote_at_offsets
from app.qa.limits import restore_sublimit_punctuation
from app.qa.schemas import Claim, Evidence
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
_APPENDIX = re.compile(r"^\s*приложени[ея]\b", re.I)
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


def _sentences(quote: LocatedQuote) -> Iterator[tuple[str, int, int]]:
    start = 0
    for boundary in [*(_m.start() for _m in _SENTENCE.finditer(quote.quote_raw)),
                     len(quote.quote_raw)]:
        s = quote.quote_raw[start:boundary]
        if s.strip():
            left = start + len(s) - len(s.lstrip())
            yield s.strip(), left, start + len(s.rstrip())
        start = boundary


def _prefer_joint_evidence(
    question: str, claims: list[Claim], located: dict[int, list[LocatedQuote]],
    sources: dict[str, ContextChunk],
) -> None:
    """Prefer a primary clause proving both parts to an unattributed appendix value."""
    if len(claims) < 2 or _COMPARE.search(question):
        return
    qterms = set(_terms(question))
    for i, claim in enumerate(claims):
        own = located[i]
        if not own or not all(_APPENDIX.match(q.chunk.scope_label or "") for q in own):
            continue
        measured = _measures(claim.text)
        if len(measured) != 1:
            continue
        stated, dimension = next(iter(measured))
        # Only substitute an original sentence; never splice a number into generated prose.
        # Two independently matching object words protect distinct items of the same unit.
        subject = (set(_terms(claim.text)) & qterms
                   & set(_terms(" ".join(q.quote_raw for q in own))))
        matches: dict[tuple[str, int, int], tuple[str, LocatedQuote]] = {}
        for j, others in located.items():
            if i == j:
                continue
            for q in others:
                if q.chunk.scope_id or q.chunk.scope_label:
                    continue
                for sentence, left, right in _sentences(q):
                    tokens = set(_terms(sentence))
                    amounts = {v for v, d in _measures(sentence) if d == dimension}
                    if (len(subject & tokens) < 2 or len((qterms & tokens) - subject) < 1
                            or len(amounts) != 1 or stated in amounts or len(sentence) > 600):
                        continue
                    offset = q.char_start - q.chunk.char_start
                    exact = quote_at_offsets(q.chunk, offset + left, offset + right)
                    matches[q.chunk.chunk_id, exact.char_start, exact.char_end] = sentence, exact
        if len(matches) == 1:
            sentence, exact = next(iter(matches.values()))
            spans = issue_evidence(sources)
            evidence_ids: list[str | Evidence] = [
                key for key, q in spans.items() if q.chunk.chunk_id == exact.chunk.chunk_id
                and q.char_start <= exact.char_start and q.char_end >= exact.char_end
            ]
            if not evidence_ids:
                continue
            claim.evidence = evidence_ids
            claim.text = " ".join(sentence.split())
            located[i] = [exact]


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
    _prefer_joint_evidence(question, claims, located, sources)
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
