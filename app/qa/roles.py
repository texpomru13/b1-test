"""Reject an explicit actor/recipient inversion in an attributed Russian list."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise

from py_rust_stemmers import SnowballStemmer

_STEMMER = SnowballStemmer("russian")
_WORD = r"[а-яё]+"
_MARKER = r"\((?:[ivxlcdm]+|[а-яa-z]|\d+)\)"
_HEADING = re.compile(rf"(?P<lead>[^:\n]{{1,240}}):\s*(?={_MARKER}\s+)", re.I)
# Limit the guard to a two-word, explicitly nominative actor followed directly
# by a finite active verb. Passive, impersonal and coordinated actors stay with
# the semantic verifier.
_CLAIM = re.compile(rf"^\s*({_WORD}(?:ый|ий|ой|ое|ее|ая|яя))\s+({_WORD})\s+"
                    rf"({_WORD}(?:ет|ит|ют|ут|ат|ят))\s+({_WORD})\b")
_ITEM = re.compile(rf"^\s*{_MARKER}\s+({_WORD})\s+({_WORD}(?:ому|ему|ой|ей))\s+"
                   rf"({_WORD})\s+({_WORD})\b", re.I)


def reversed_list_actor(claim: str, quotes: Iterable[str]) -> str | None:
    """Return the cited lead only for an explicit dative recipient promoted to actor."""
    def clean(text: str) -> str:
        return text.casefold().replace("ё", "е").replace("\u00ad\n", "").strip()

    match = _CLAIM.match(clean(claim))
    if match is None:
        return None
    adjective, noun, verb, obj = match.groups()
    actor = tuple(_STEMMER.stem_word(word) for word in (adjective, noun))
    direct = re.compile(rf"\b{adjective}\s+{noun}\s+{verb}\b")
    passive = re.compile(rf"\b{verb}(?:ся|сь)\b")
    lead = None
    for quote in quotes:
        text = clean(quote)
        # Independently attributed reciprocal actions require semantic review.
        if direct.search(text) or passive.search(text):
            return None
        for heading in _HEADING.finditer(text):
            intro = heading["lead"]
            intro_words = [_STEMMER.stem_word(word) for word in re.findall(_WORD, intro)]
            if actor in pairwise(intro_words):
                return None
            for part in text[heading.end():].split(";"):
                if not re.match(rf"^\s*{_MARKER}\s+", part, re.I):
                    break
                item = _ITEM.match(part)
                if item is None:
                    continue
                action, adj, recipient, object_word = item.groups()
                if (verb == action and obj == object_word
                        and actor == (_STEMMER.stem_word(adj), _STEMMER.stem_word(recipient))):
                    lead = intro.strip()
    return lead
