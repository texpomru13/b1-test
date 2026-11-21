"""Resolve an unqualified time unit against a definition in the document."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.search import ContextChunk

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


def attach_day_definitions(question: str, units: dict[str, list[ContextChunk]]) -> None:
    if not _ASKS_PERIOD.search(question):
        return
    definitions = [
        (unit, match["kind"].lower()[:3])
        for unit in units.values()
        for match in DAY_DEFINITION.finditer(unit[-1].text)
    ]
    for key, unit in units.items():
        anchor = unit[-1]
        if not DURATION.search(anchor.text) or _EXPLICIT.search(anchor.text):
            continue
        local = [(parts, kind) for parts, kind in definitions
                 if parts[-1].scope_id and parts[-1].scope_id == anchor.scope_id]
        relevant = local or [(parts, kind) for parts, kind in definitions
                             if parts[-1].scope_id is None]
        if len({kind for _, kind in relevant}) == 1:
            definition, _ = min(relevant, key=lambda item: item[0][-1].chunk_index)
            units[key] = [*definition, *unit]
