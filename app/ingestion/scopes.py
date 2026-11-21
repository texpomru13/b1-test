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
_APPENDIX = re.compile(
    r"^[ \t]*приложение[ \t]*(?:№[ \t]*)?(\d+(?:\.\d+)*)\b"
    r"(?=[ \t]*$|[ \t]*[.:—–-]|[ \t]+к\b|[ \t]*[«\"]|[ \t]+(?-i:[А-ЯЁA-Z]))"
    r"[^\n]{0,100}$", re.I,
)
_COMMON = re.compile(r"^\s*(?:\d+[.)]?\s*)?общие (?:условия|положения|правила)"
                     r"\s+(?:для\s+)?всех\b", re.I)


@dataclass(frozen=True)
class DocumentScope:
    scope_id: str
    scope_label: str
    scope_label_span: tuple[int, int]
    appendix_number: str | None = None


def scope_heading(text: str, start: int, line: str) -> DocumentScope | None:
    """An offer's title may occupy several PDF lines. Never search into its clauses."""
    match: re.Match[str] | None
    if line.strip().casefold() == "публичная оферта":
        title = "\n".join(text[start:start + 450].splitlines()[:4])
        match = _NAME.search(title)
        if not match:
            left = start + len(line) - len(line.lstrip())
            return DocumentScope(f"offer-{start}", line.strip(), (left, left + len(line.strip())))
    elif (appendix := _APPENDIX.match(line)):
        label = line.strip()
        left = start + line.index(label)
        return DocumentScope(f"appendix-{start}", label, (left, left + len(label)),
                             appendix[1])
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


def scope_ancestors(
    text: str, regions: list[tuple[DocumentScope, int, int]],
) -> dict[str, tuple[str, ...]]:
    """Resolve links within this document. A reused number needs an explicit owner."""
    scopes = {scope.scope_id: scope for scope, _, _ in regions}
    appendices: dict[str, list[DocumentScope]] = {}
    names: dict[str, list[DocumentScope]] = {}
    for scope in scopes.values():
        if scope.appendix_number:
            appendices.setdefault(scope.appendix_number, []).append(scope)
        else:
            names.setdefault(scope_key(scope.scope_label), []).append(scope)
    owners: dict[str, set[str]] = {}
    parents: dict[str, set[str]] = {key: set() for key in scopes}
    for group in appendices.values():
        for child in group:
            for owner_name in _EXPLICIT.finditer(child.scope_label):
                child_owners = owners.setdefault(child.scope_id, set())
                matches = names.get(scope_key(owner_name[1]), [])
                if len(matches) == 1:
                    child_owners.add(matches[0].scope_id)
                    parents[child.scope_id].add(matches[0].scope_id)
    for source, start, end in regions:
        for ref in _APPENDIX_REF.finditer(text, start, end):
            if source.scope_label_span[0] <= ref.start() < source.scope_label_span[1]:
                continue
            children = appendices.get(ref[1], [])
            local = [child for child in children if source.scope_id in owners.get(
                child.scope_id, set())]
            # An appendix assigned to a different offer cannot be inherited by mention alone.
            compatible = [child for child in children if child.scope_id not in owners]
            candidates = local or compatible
            if len(candidates) == 1 and candidates[0].scope_id != source.scope_id:
                parents[candidates[0].scope_id].add(source.scope_id)

    result = {}
    for key in scopes:
        seen = {key}
        pending = list(parents[key])
        while pending:
            parent = pending.pop()
            if parent not in seen:
                seen.add(parent)
                pending.extend(parents[parent] - seen)
        result[key] = tuple(scopes[p].scope_label for p in scopes if p in seen and p != key)
    return result


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


def requested_branches(
    requested: set[str], paths: set[tuple[str, ...]],
) -> set[str]:
    """An offer and its named appendix in the same question describe one branch."""
    ancestors = {ancestor for key, *parents in paths if key in requested
                 for ancestor in parents if ancestor != key}
    return requested - ancestors or requested


def belongs_to_request(path: tuple[str, ...], requested: set[str], branches: set[str]) -> bool:
    return (not requested or not path or path[0] in requested
            or not branches.isdisjoint(path))
