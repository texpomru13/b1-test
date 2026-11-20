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


def detect_sections(canonical_text: str) -> list[SectionSpan]:
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in canonical_text.split("\n"):
        if 0 < len(line) <= MAX_HEADING_CHARS:
            label = _section_label(line)
            if label:
                headings.append((offset, label[:MAX_LABEL_CHARS]))
        offset += len(line) + 1  # +1 for the split "\n"

    sections: list[SectionSpan] = []
    for index, (start, label) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(canonical_text)
        sections.append(SectionSpan(label=label, char_start=start, char_end=end))
    return sections


def section_at(sections: list[SectionSpan], offset: int) -> str | None:
    for span in sections:
        if span.char_start <= offset < span.char_end:
            return span.label
    return None
