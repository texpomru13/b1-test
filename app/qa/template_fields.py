"""Preserve a visibly unfilled duration in source templates."""
from __future__ import annotations

import re

_PREP = r"за|через|в\s+течение"
_BOUND = r"(?:не\s+(?:менее|более|раньше|позднее)\s+чем\s+)?"
_BLANK = re.compile(rf"\b(?P<prep>{_PREP})\s+_{{2,}}", re.I)
_VAGUE = re.compile(
    rf"\b{_BOUND}(?P<prep>{_PREP})\s+"
    r"(?:определ[её]нн|установленн|фиксированн|оговор[её]нн|заданн|конкретн)\w*\s+"
    r"(?:срок|период)\w*\b", re.I,
)
_DURATION = re.compile(
    r"\b(?:\d+|одн\w*|дв[аеух]\w*|тр[иеё]\w*|четыр\w*|"
    r"пят\w*|шест\w*|сем\w*|вос\w*|девя\w*|десят\w*)"
    r"(?:\s*\([^)]*\))?\s+(?:(?:рабоч|календарн|банковск)\w*\s+)?"
    r"(?:минут|час|дн|день|сут|недел|месяц|лет|год)\w*\b", re.I,
)


def clarify_unfilled_duration(text: str, quotes: list[str]) -> str:
    """Replace a claimed 'specified period' only when its own slot is blank."""
    vague = _VAGUE.search(text)
    if vague is None:
        return text
    prep = " ".join(vague["prep"].lower().split())
    source = next((q for q in quotes if any(
        " ".join(m["prep"].lower().split()) == prep for m in _BLANK.finditer(q)
    )), None)
    if source is None or any(_DURATION.search(q) for q in quotes):
        return text
    # Use the clause itself: dropping only the vague phrase can break grammar.
    clause = " ".join(source.split()).rstrip(".")
    return f"{clause}. Поле для указания срока в этом пункте не заполнено."
