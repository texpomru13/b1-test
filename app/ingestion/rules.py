"""Keep the complete text of a conditional permission or prohibition together."""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import pairwise

_SENTENCE_END = re.compile(
    r'(?:(?<=[а-яёa-z]{3}\.)|(?<=[)»]\.)|(?<=[!?]))\s+(?=[А-ЯЁA-Z\d])'
    r'|\n[ \t]*\n+'
)
_MODAL = re.compile(r"\b(?:запрещ|запрет\b|вправе|разреш|допуска[ею]|нельзя|"
                    r"можно|имеет прав[оа])", re.I)
COMMITMENT = re.compile(r"\b(?:гарантир|обязуе|обязую|обязан)", re.I)
_LIMIT = re.compile(r"\b(?:за исключением|в порядке исключения|исключение|"
                    r"кроме|только|исключительно|при условии|"
                    r"при наличии|по согласованию|с согласия|если)\b", re.I)
_PAREN_ITEM = re.compile(r"^[ \t]*\(([1-9]\d{0,2})\)[ \t]+(?=\S)", re.M)
_CONTINUES_RULE = re.compile(r"(?:кроме|за исключением|однако|при этом|если|"
                             r"при условии|за вычетом)\b", re.I)
_EXCEPTION = re.compile(r"(?:однако[,\s]+)?(?:кроме|за исключением|"
                        r"в порядке исключения|исключение)\b", re.I)


def parenthesized_items(text: str) -> Iterator[int]:
    """Coordinates of standalone numbered clauses, keeping a colon-led list together."""
    previous = 0
    last = 0
    inherits_lead = False
    markers = list(_PAREN_ITEM.finditer(text))
    for index, marker in enumerate(markers):
        number = int(marker[1])
        sibling = ((previous and number == previous + 1)
                   or (index + 1 < len(markers) and int(markers[index + 1][1]) == number + 1))
        if number != previous + 1 or not previous:
            inherits_lead = text[last:marker.start()].rstrip().endswith(":")
        previous = number
        last = marker.end()
        # Lowercase items and continuations can depend on the preceding norm.
        if (sibling and not inherits_lead and text[last].isupper()
                and not _CONTINUES_RULE.match(text, last)):
            yield marker.start()


def conditional_rules(text: str) -> Iterator[tuple[int, int]]:
    boundaries = sorted({0, len(text), *parenthesized_items(text),
                         *(m.end() for m in _SENTENCE_END.finditer(text)
                           if not _EXCEPTION.match(text, m.end()))})
    for start, end in pairwise(boundaries):
        raw = text[start:end]
        if _LIMIT.search(raw) and (_MODAL.search(raw) or COMMITMENT.search(raw)):
            yield start + len(raw) - len(raw.lstrip()), start + len(raw.rstrip())
