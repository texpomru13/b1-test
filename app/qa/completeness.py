"""Keep qualifications attached to the cited rule, before semantic verification."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from py_rust_stemmers import SnowballStemmer

from app.ingestion.scopes import scope_key
from app.qa.schemas import Claim
from app.retrieval.day_definition import DAY_DEFINITION, DURATION
from app.retrieval.search import ContextChunk

if TYPE_CHECKING:
    from app.qa.grounding import LocatedQuote

_STEMMER = SnowballStemmer("russian")
_WORD = re.compile(r"[а-яёa-z]{3,}|\d+(?:[.,]\d+)?", re.I)
_STOP = set("что это как для его при этом или того если либо также этот может согласно "
            "договор документ пункт вопрос составлять указать".split())
_QUALIFIED = re.compile(
    r"\b(?:за исключением|по требованию|без ущерба|при условии|в том числе|в ином случае|"
    r"если|начиная с|с момента|со дня|до (?:полного|окончательного)|после чего|а затем|"
    r"в письменной форме|письменн\w*\s+(?:уведомлен|заявлен)|кажд\w*\s+(?:понедельник|"
    r"вторник|сред|четверг|пятниц|суббот|воскресен)|состоит из|определение включает)", re.I)
_SPLIT = re.compile(r"\n\s*\n|(?<=[а-яё]{3}\.)\s+(?=[А-ЯЁ])|(?<=[!?;])\s+(?=[А-ЯЁ])"
                    r"|\n(?=\s*(?:СТАТЬЯ|Статья)\s+\d)")
_PREFIX = re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s+|Приложение\s*\n)", re.I)
_PAREN = re.compile(r"\((если\s+[^()]{10,220})\)", re.I)
_DAY_KIND = re.compile(r"\b(рабоч|календарн)\w*\s+(?:дн\w*|день)\b", re.I)


@lru_cache(maxsize=4096)
def _stem(word: str) -> str:
    return _STEMMER.stem_word(word)


def words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower().replace("ё", "е")) if w not in _STOP}


def _joined_quotes(quotes: list[LocatedQuote]) -> Iterable[str]:
    """Rejoin only spans from the same source with an exactly known whitespace gap."""
    ordered = sorted(quotes, key=lambda q: (q.chunk.chunk_id, q.char_start))
    previous: LocatedQuote | None = None
    start = 0
    for q in ordered:
        if previous and (q.chunk.chunk_id != previous.chunk.chunk_id or
                         q.char_start < previous.char_end or
                         q.chunk.text[previous.char_end-q.chunk.char_start:
                                      q.char_start-q.chunk.char_start].strip()):
            yield previous.chunk.text[start:previous.char_end-previous.chunk.char_start]
            previous = None
        if previous is None:
            start = q.char_start - q.chunk.char_start
        previous = q
    if previous:
        yield previous.chunk.text[start:previous.char_end-previous.chunk.char_start]


def _complete_wording(text: str, quotes: list[LocatedQuote]) -> str:
    terms = words(text)
    if len(terms) < 2:
        return text
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    matches: dict[str, int] = {}
    provisos: set[str] = set()
    for raw in _joined_quotes(quotes):
        for match in _PAREN.finditer(raw):
            # A renderer may insert a blank line in the middle of a parenthesis.
            # Bind it to the immediately preceding assertion, not a nearby clause.
            lead = _SPLIT.split(raw[:match.start()])[-1]
            shared = len(terms & words(lead))
            if (shared >= 3 and shared >= len(terms) / 2
                    and numbers <= set(re.findall(r"\d+(?:[.,]\d+)?", lead))):
                provisos.add(" ".join(match[1].split()))
        for sentence in _SPLIT.split(raw):
            sentence = _PREFIX.sub("", sentence.strip(), count=1)
            sentence = " ".join(sentence.split())
            if not _QUALIFIED.search(sentence) or not numbers <= set(re.findall(
                    r"\d+(?:[.,]\d+)?", sentence)):
                continue
            source_terms = words(sentence)
            overlap = len(terms & source_terms)
            if overlap < 2 or overlap < len(terms) * 0.64 or len(source_terms - terms) < 2:
                continue
            # A selected excerpt may end mid-sentence. A complete parenthetical
            # proviso can still be preserved without copying that dangling ending.
            if (20 <= len(sentence) <= 850 and sentence[0].isupper()
                    and sentence.endswith((".", ";", "!", "?"))
                    and not sentence.endswith("…") and not re.search(r"\b\d+[.)]$", sentence)
                    and not re.search(r"\b\d+(?:\.\d+)+\.\s+[А-ЯЁ]", sentence)):
                matches[sentence.rstrip(";")] = overlap
            elif m := _PAREN.search(sentence):
                provisos.add(m[1])
    if matches:
        best = max(matches.values())
        unique = [s for s, score in matches.items() if score == best]
        if len(unique) == 1:
            return unique[0]
    if len(provisos) == 1 and not _PAREN.search(text) and "если" not in text.lower():
        return text.rstrip(". ") + " (" + provisos.pop() + ")."
    return text


def preserve_qualifications(claims: list[Claim], located: dict[int, list[LocatedQuote]]) -> None:
    for index, claim in enumerate(claims):
        quotes = located[index]
        wording = _complete_wording(claim.text, quotes)
        if wording == claim.text:
            continue
        # A paraphrase can intentionally name the offer above its price row.
        # Keep that label when the selected sentence references "this offer".
        labels = {q.chunk.scope_label for q in quotes if q.chunk.scope_label}
        named = [label for label in labels if scope_key(label) in scope_key(claim.text)
                 and scope_key(label) not in scope_key(wording)]
        if len(named) <= 1:
            response = f"«{named[0]}». {wording}" if named else wording
            direct = re.match(r"^(Да|Нет)[.,]\s", claim.text, re.I)
            claim.text = f"{direct[1]}. {response}" if direct else response


def add_defined_day(claim: Claim, quotes: list[LocatedQuote],
                    sources: Iterable[ContextChunk]) -> None:
    """A bare day count inherits an explicit glossary entry unless locally qualified."""
    periods = set(re.findall(r"\b\d+\b", claim.text))
    if not periods or not re.search(r"\b(?:день|дня|дней|дни)\b", claim.text, re.I):
        return
    if not any(m[1] in periods for q in quotes for m in DURATION.finditer(q.quote_raw)):
        return
    if any(_DAY_KIND.search(q.quote_raw) for q in quotes if not DAY_DEFINITION.search(q.quote_raw)):
        return
    scopes = {q.chunk.scope_id for q in quotes}
    definitions = [(chunk, m) for chunk in sources if not chunk.scope_id or chunk.scope_id in scopes
                   for m in DAY_DEFINITION.finditer(chunk.text)]
    kinds = {m["kind"].lower()[:3] for _, m in definitions}
    if len(kinds) != 1:
        return
    chunk, match = definitions[0]
    from app.qa.grounding import quote_at_offsets

    proof = quote_at_offsets(chunk, match.start(), match.end())
    if not any(q.chunk.chunk_id == chunk.chunk_id and q.char_start <= proof.char_start
               and q.char_end >= proof.char_end for q in quotes):
        quotes.append(proof)
    if not _DAY_KIND.search(claim.text):
        unit = "рабочие" if match["kind"].lower().startswith("раб") else "календарные"
        claim.text = (f"{claim.text.rstrip('. ')}. "
                      f"По определению документа срок исчисляется в {unit} дни.")
