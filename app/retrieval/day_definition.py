"""Resolve an unqualified time unit against a definition in the document."""

from __future__ import annotations

import re
from typing import Any

# A definition names the term on the left. A deadline that merely uses
# "рабочие дни" elsewhere in the document is not a definition.
DAY_DEFINITION = re.compile(
    r'^[ \t]*(?:[«"]?день[»"]?(?:\s*[,/]\s*дни)?|дни)(?:[ \t]+'
    r'(?:означа\w*|понимаются\s+как))?[ \t]*(?:[—–:=\-][ \t]*)?'
    r'(?P<kind>рабоч(?:ие|ий)|календарн(?:ые|ый))\s+(?:дни|день)\b', re.I | re.M,
)
DURATION = re.compile(r'\b(\d+)\s*(?:\([^()\d]{1,45}\)\s*)?(?:день|дня|дней|дни)\b', re.I)
_EXPLICIT = re.compile(r'\b(?:рабоч|календарн)\w*\s+д(?:н|ень)', re.I)
_ASKS_PERIOD = re.compile(r'\b(?:срок\w*|когда|дней|дня|времени|период\w*)\b', re.I)


def prioritize_day_definition(question: str, ordered: list[Any]) -> list[Any]:
    if not _ASKS_PERIOD.search(question) or not ordered:
        return ordered
    primary = ordered[0].payload or {}
    text = primary.get("text", "")
    if not DURATION.search(text) or _EXPLICIT.search(text):
        return ordered
    definitions = []
    for item in ordered:
        payload = item.payload or {}
        if payload.get("scope_id") not in (None, primary.get("scope_id")):
            continue
        match = DAY_DEFINITION.search(payload.get("text", ""))
        if match:
            definitions.append((item, match["kind"].lower()[:3]))
    if not definitions or len({kind for _, kind in definitions}) != 1:
        return ordered
    definition = definitions[0][0]
    if definition in ordered[:2]:
        return ordered
    return [ordered[0], definition, *(p for p in ordered[1:] if p.id != definition.id)]
