"""Section heading detector for Russian contracts.

- A heading line starts (after up to 3 spaces) with either
  * a clause number `4`, `4.2`, `4.2.1` (dot-separated, <= 4 components,
    optionally followed by `.` or `)`) AND a title starting with an UPPERCASE
    letter — so `0,1%` (comma), `4.20 рублей` (lowercase continuation) and bare
    amounts never match, or
  * `Приложение` with an optional `№`/number (end of line allowed).
- The heading line must be at most MAX_HEADING_CHARS long.
- Headings retain their original labels. A line introducing a numbered
  obligation uses the clause number. The preamble has section=None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.scopes import DocumentScope, common_heading, scope_heading

MAX_HEADING_CHARS = 150
MAX_LABEL_CHARS = 200

_NUM_HEADING = re.compile(
    r"^\s{0,3}(?P<num>\d{1,3}(?:\.\d{1,3}){0,3})[.)]?\s+(?=[A-ZА-ЯЁ«\"])"
)
_APP_HEADING = re.compile(r"^\s{0,3}Приложение(?:\s*№?\s*\d+)?\s*(?:$|[:.\s])")


def _section_label(line: str) -> str | None:
    if _APP_HEADING.match(line):
        return line.strip()
    match = _NUM_HEADING.match(line)
    if not match:
        return None
    title = line[match.end():].strip()
    # A numbered obligation often wraps onto the next PDF line. Keep its
    # actual clause number for navigation instead of presenting a sentence
    # fragment as the name of a section.
    if (len(title.split()) > 8 or title.endswith((",", ";", ":"))
            or _CLAUSE_VERB.search(title)):
        return match["num"]
    return line.strip()


_CLAUSE_VERB = re.compile(
    r"\b(?:обязуется|обязаны?|долж(?:ен|на|ны)|вправе|уплачива\w*|"
    r"составля(?:ет|ют)|возмеща\w*|нес[её]т|предоставляет|переда[её]т)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionSpan:
    label: str
    char_start: int  # heading line start in canonical text
    char_end: int  # exclusive; end of the section (next heading or EOF)
    scope: DocumentScope | None = None
    clause_id: int | None = None
    parent_clause_id: int | None = None
    governing_clause_ids: tuple[int, ...] = ()


def detect_sections(canonical_text: str) -> list[SectionSpan]:
    headings: list[tuple[int, str, DocumentScope | None, str]] = []
    scope: DocumentScope | None = None
    offset = 0
    for line in canonical_text.splitlines(keepends=True):
        raw = line.removesuffix("\n")
        named = scope_heading(canonical_text, offset, raw)
        label = _section_label(raw) if len(raw) <= MAX_HEADING_CHARS else None
        if named or common_heading(raw):
            scope = named
            label = raw.strip()
        # Dotted subclauses can begin with a lowercase infinitive. A terminal
        # period distinguishes the number from an amount such as "4.20 рублей".
        clause = re.match(r"^[ \t]*(\d{1,3}(?:\.\d{1,3}){0,5})[.)]?[ \t]+", raw)
        if not label and clause and clause[0].rstrip().endswith((".", ")")):
            label = clause[1]
        if label:
            headings.append((offset, label[:MAX_LABEL_CHARS], scope, clause[1] if clause else ""))
        offset += len(line)

    sections: list[SectionSpan] = []
    # Path elements are (number, source position, introduces a list).
    path: list[tuple[str, int, bool]] = []
    for index, (start, label, owner, number) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(canonical_text)
        if not index or headings[index - 1][2] != owner or not number:
            path.clear()
        while path and not number.startswith(path[-1][0] + "."):
            path.pop()
        sections.append(SectionSpan(
            label, start, end, owner, start if number else None,
            path[-1][1] if path else None, tuple(pos for _, pos, lead in path if lead),
        ))
        if number:
            path.append((number, start, canonical_text[start:end].rstrip().endswith(":")))
    return sections


def section_at(sections: list[SectionSpan], offset: int) -> str | None:
    for span in sections:
        if span.char_start <= offset < span.char_end:
            return span.label
    return None
