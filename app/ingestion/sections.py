"""Section heading detector for Russian contracts.

Documented heuristics (plan §5.2: boundaries matter more than even length, and a
decimal amount must not be mistaken for a clause number without line context):

- A heading line starts (after up to 3 spaces) with either
  * a clause number `4`, `4.2`, `4.2.1` (dot-separated, <= 4 components,
    optionally followed by `.` or `)`) AND a title starting with an UPPERCASE
    letter — so `0,1%` (comma), `4.20 рублей` (lowercase continuation) and bare
    amounts never match, or
  * `Приложение` with an optional `№`/number (end of line allowed).
- The heading line must be at most MAX_HEADING_CHARS long.
- The section label is the trimmed heading line; text before the first heading
  has section=None (never invented).
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


def _is_heading(line: str) -> bool:
    return bool(_NUM_HEADING.match(line) or _APP_HEADING.match(line))


@dataclass(frozen=True)
class SectionSpan:
    label: str
    char_start: int  # heading line start in canonical text
    char_end: int  # exclusive; end of the section (next heading or EOF)


def detect_sections(canonical_text: str) -> list[SectionSpan]:
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in canonical_text.split("\n"):
        stripped = line.strip()
        if 0 < len(line) <= MAX_HEADING_CHARS and _is_heading(line):
            headings.append((offset, stripped[:MAX_LABEL_CHARS]))
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
