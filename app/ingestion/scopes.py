"""Source-anchored names of independent offers and appendices inside a document."""

from __future__ import annotations

import re
from dataclasses import dataclass

_QUOTED = r'[«"]([^»"\n]{1,160}(?:\n[^»"\n]{1,100})?)[»"]'
_NAME = re.compile(r"\bакции\s+" + _QUOTED, re.I)
_NAMED_HEADING = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:маркетинговая\s+)?"
    r"(?:акция|тариф|программа|предложение|объект)\s+" + _QUOTED + r"\s*[.:]?\s*$", re.I,
)
_APPENDIX = re.compile(r"^\s*приложение\s*(?:№\s*)?\d+\b[^\n]{0,100}$", re.I)
_COMMON = re.compile(r"^\s*(?:\d+[.)]?\s*)?общие (?:условия|положения|правила)"
                     r"\s+(?:для\s+)?всех\b", re.I)


@dataclass(frozen=True)
class DocumentScope:
    scope_id: str
    scope_label: str
    scope_label_span: tuple[int, int]


def scope_heading(text: str, start: int, line: str) -> DocumentScope | None:
    """An offer's title may occupy several PDF lines. Never search into its clauses."""
    match: re.Match[str] | None
    if line.strip().casefold() == "публичная оферта":
        title = "\n".join(text[start:start + 450].splitlines()[:4])
        match = _NAME.search(title)
        if not match:
            left = start + len(line) - len(line.lstrip())
            return DocumentScope(f"offer-{start}", line.strip(), (left, left + len(line.strip())))
    elif _APPENDIX.match(line):
        label = line.strip()
        left = start + line.index(label)
        return DocumentScope(f"appendix-{start}", label, (left, left + len(label)))
    else:
        match = _NAMED_HEADING.match(line)
    if not match:
        return None
    left, right = match.span(1)
    return DocumentScope(f"offer-{start}", match[1], (start + left, start + right))


def common_heading(line: str) -> bool:
    return bool(_COMMON.match(line))


def scope_key(label: str) -> str:
    # Match case and whitespace independently, preserve digits and full word boundaries.
    return " ".join(re.findall(r"[\w]+", label.casefold().replace("ё", "е")))


_EXPLICIT = re.compile(r"\b(?:акци[яиюй]|акций|тариф[аеу]?|программ[аеуы]|"
                       r"предложени[еяиюй]|объект[аеу]?)\s+" + _QUOTED, re.I)
_APPENDIX_REF = re.compile(r"\bприложени[еяиюй]\s*(?:№\s*)?(\d+(?:\.\d+)*)\b", re.I)


def requested_scope_keys(text: str, labels: set[str]) -> set[str]:
    known = {scope_key(label) for label in labels}
    keys = known.intersection(scope_key(m[1]) for m in re.finditer(_QUOTED, text))
    keys.update(scope_key(match[1]) for match in _EXPLICIT.finditer(text))
    # A longer name may contain the entire title of a neighbouring offer.
    # Reserve quoted names and match the longest remaining names first.
    outside = " " + scope_key(re.sub(_QUOTED, " ", text)) + " "
    for label in sorted(known, key=len, reverse=True):
        if len(label) > 1 and f" {label} " in outside:
            keys.add(label)
            outside = outside.replace(f" {label} ", " ")
    for ref in _APPENDIX_REF.finditer(text):
        prefix = "приложение " + scope_key(ref[1])
        matches = {scope_key(label) for label in labels
                   if (scope_key(label) + " ").startswith(prefix + " ")}
        keys.update(matches or {prefix})
    return keys
