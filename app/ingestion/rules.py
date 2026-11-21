"""Keep the complete text of a conditional permission or prohibition together."""

from __future__ import annotations

import re
from collections.abc import Iterator

_SENTENCE_END = re.compile(
    r'(?:(?<=[а-яёa-z]{3}\.)|(?<=[)»]\.)|(?<=[!?]))\s+(?=[А-ЯЁA-Z\d])'
    r'|\n[ \t]*\n+'
)
_MODAL = re.compile(r"\b(?:запрещ|запрет\b|не\s+вправе|вправе|разреш|допуска[ею]|нельзя)", re.I)
_LIMIT = re.compile(r"\b(?:за исключением|кроме|только|исключительно|при условии|"
                    r"при наличии|по согласованию|с согласия|если)\b", re.I)


def conditional_rules(text: str) -> Iterator[tuple[int, int]]:
    start = 0
    for end in [*(m.end() for m in _SENTENCE_END.finditer(text)), len(text)]:
        raw = text[start:end]
        if _MODAL.search(raw) and _LIMIT.search(raw):
            yield start + len(raw) - len(raw.lstrip()), start + len(raw.rstrip())
        start = end
