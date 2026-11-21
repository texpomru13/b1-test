"""Keep a cited designation when an optional explanatory gloss has no source."""

from __future__ import annotations

import re
from collections.abc import Sequence

from py_rust_stemmers import SnowballStemmer

_STEMMER = SnowballStemmer("russian")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_TERM = r"[А-ЯЁA-Z][А-ЯЁA-Z0-9]{1,10}"
_ACRONYM = re.compile(rf"(?<!\w)(?P<term>{_TERM})\s*$")
_REF = re.compile(r"(?<!\w)(?:пп?\.|подпункт\w*|пункт\w*|ст\.|стать[яеиюёй]+)\s*"
                  r"(?P<term>\d+(?:\.\d+)*|[а-яa-z]\))"
                  r"(?:\s*(?:,|или|и)\s*(?:[а-яa-z]\)|\d+(?:\.\d+)*))*\s*$", re.I)
_PAREN = re.compile(r"\((?P<gloss>[^()\n]{2,320})\)")
_CONNECTOR = re.compile(r"\s*(?:[—–:-]|,?\s*(?:то есть|т\.\s*е\.|это)\s*)\s*", re.I)
_MEANING = re.compile(r"\b(?:расшифр\w*|означа\w*|обознача\w*|подразумева\w*|"
                      r"что\s+понимают|какие\s+именно|что\s+(?:включ|перечисл))", re.I)
_QUALIFIER = re.compile(r"^(?:если|кроме|без|при\s+условии|за\s+исключением|"
                        r"не\s+(?:более|менее|раньше|позже))\b", re.I)


def _terms(text: str) -> list[str]:
    return [_STEMMER.stem_word(w.casefold().replace("ё", "е")) for w in _TOKEN.findall(text)]


def _prefix_terms(source: str, gloss: list[str]) -> bool:
    return _terms(source)[:len(gloss)] == gloss


def _defined(term: str, gloss: str, quotes: Sequence[str]) -> bool:
    wanted = _terms(gloss)
    marker = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")
    for quote in quotes:
        for m in marker.finditer(quote):
            suffix = quote[m.end():].lstrip()
            after = _CONNECTOR.match(suffix)
            if suffix.startswith("(") and _prefix_terms(suffix[1:], wanted):
                return True
            if after and _prefix_terms(suffix[after.end():], wanted):
                return True
            # A definition is also often printed in the opposite order:
            # «расширенное название (далее — АБ)».
            before = re.sub(r"\(\s*(?:далее\s*(?:по тексту\s*)?[—–:-]?\s*)?$", "",
                            quote[:m.start()], flags=re.I)
            if before != quote[:m.start()] and _terms(before)[-len(wanted):] == wanted:
                return True
    return False


_REVERSE = re.compile(rf"\((?P<term>{_TERM})\)")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁ](?:[а-яё]|\s+[а-яё]))|\n\s*\n")


def _printed_rule(text: str, term: str, quotes: Sequence[str]) -> str | None:
    marker = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")
    draft = set(_terms(text))
    matches: dict[str, int] = {}
    for quote in quotes:
        for raw in _SENTENCE.split(quote):
            source = re.sub(r"^\s*\d+(?:\.\d+)*[.)]\s*", "", " ".join(raw.split()))
            if not 16 <= len(source) <= 700 or not marker.search(source):
                continue
            source_terms = set(_terms(source))
            common = len(draft & source_terms)
            if common >= 3 and common >= len(source_terms) * .65 and common >= len(draft) * .5:
                matches[source] = common
    if matches:
        best = max(matches.values())
        selected = [s for s in matches if matches[s] == best]
        if len(selected) == 1:
            return selected[0]
    return None


_EXPLICIT = re.compile(rf"(?<!\w)(?P<term>{_TERM})\s*,\s*(?:то есть|т\.\s*е\.)\s+"
                       r"(?P<gloss>[А-Яа-яЁё][^,;.!?()]{2,150})(?=[.;!?]?$)")


def without_unproven_gloss(text: str, quotes: Sequence[str], question: str) -> str:
    """Uncited expansions of a designation cannot create a new kind of covered event."""
    for reverse in _REVERSE.finditer(text):
        term = reverse["term"]
        if not any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", q) for q in quotes):
            continue
        preceding = re.split(r"[,;:!?()—–]|\b(?:в|на|из|для|при|через|согласно|"
                             r"означает|называется)\s+", text[:reverse.start()], flags=re.I)[-1]
        if preceding.strip() and _defined(term, preceding, quotes):
            continue
        source = _printed_rule(text, term, quotes)
        if source and not _MEANING.search(question):
            return source
        raise ValueError(f"Источник не связывает раскрытое название с «{term}». "
                         "Выбери определение или приведи применимое положение дословно.")
    # A terminal «то есть …» is also an explicit apposition. Preserve any
    # confirmed spelling and route an ungrounded one through the same policy.
    terminal = _EXPLICIT.search(text)
    if (terminal and any(re.search(r"(?<!\w)" + re.escape(terminal["term"]) + r"(?!\w)", q)
                         for q in quotes)
            and not _defined(terminal["term"], terminal["gloss"], quotes)):
        text = (text[:terminal.start()] + terminal["term"] + " (" + terminal["gloss"] + ")"
                + text[terminal.end():])
    for m in reversed(list(_PAREN.finditer(text))):
        gloss = m["gloss"].strip()
        if not re.search(r"[а-яё]{2,}", gloss, re.I) or _QUALIFIER.match(gloss):
            continue
        prefix = text[:m.start()]
        label = _ACRONYM.search(prefix)
        reference = _REF.search(prefix)
        if not (label or reference):
            continue
        if label:
            term = label["term"]
            if _defined(term, gloss, quotes):
                continue
            printed = any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", q) for q in quotes)
        else:
            assert reference is not None
            term = reference["term"]
            # A verbatim explanation carried by the cited provision stays in the answer.
            if any(_terms(gloss) == _terms(p["gloss"]) for q in quotes for p in _PAREN.finditer(q)):
                continue
            printed = any(re.search(r"(?<!\w)" + re.escape(term) + r"(?![\w.]*\d)", q, re.I)
                          for q in quotes)
        if not printed:
            continue
        remainder = prefix.rstrip() + text[m.end():]
        if _MEANING.search(question) or _terms(remainder) == _terms(term):
            raise ValueError(f"Для расшифровки «{term}» нужна цитата с определением "
                             "или полный текст соответствующего подпункта с условиями.")
        text = remainder
    return text
