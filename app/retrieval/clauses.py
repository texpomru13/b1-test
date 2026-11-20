"""Recover governing clauses from consecutively numbered list items in the document."""

from __future__ import annotations

import re

from app.inference.contract import RerankDocument

_GOVERNING_WORD = re.compile(
    r"\b(?:прав[ао]\b|вправе|разреш|запрещ|допуска|обязан|обязу|"
    r"может|могут|можно|нельзя|если|при услови|в случае|в следующих случаях|за исключением)",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"^[ \t]*(\d+(?:\.\d+){0,5})[.)]?[ \t]+(?=\S)", re.MULTILINE)


def _lead(text: str) -> bool:
    # A sentence ending in a colon introduces its list. Other numbered headings
    # (including a complete neighbouring obligation) cannot supply that meaning.
    return text.rstrip().endswith(":") and bool(_GOVERNING_WORD.search(text))


def local_list_lead(text: str, offset: int) -> str | None:
    """Exact introductory span when both the lead and the item are in a single chunk."""
    headings = [match for match in _CLAUSE.finditer(text) if match.start() <= offset]
    if len(headings) < 2:
        return None
    child = headings[-1][1]
    parent_number = child.rpartition(".")[0]
    for index in range(len(headings) - 2, -1, -1):
        parent = headings[index]
        if parent[1] == parent_number:
            lead = text[parent.start():headings[index + 1].start()].strip()
            return lead if _lead(lead) else None
        if not parent[1].startswith(parent_number + "."):
            break
    return None


def list_rerank_documents(
    documents: list[RerankDocument], positions: dict[str, int],
) -> tuple[list[RerankDocument], dict[str, str]]:
    """Score a list heading with its subject matter and items with the governing text.

    Every score belongs to an original fused candidate. Only consecutive source
    chunks can prove that a heading still governs the item across page breaks.
    The actual generator sources and citation coordinates remain verbatim.
    """
    parents: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    texts = {doc.id: doc.text for doc in documents}
    stack: list[tuple[str, RerankDocument]] = []
    previous_position = -2
    for doc in sorted(documents, key=lambda item: positions[item.id]):
        match = _CLAUSE.match(doc.text)
        number = match[1] if match else ""
        position = positions[doc.id]
        if position != previous_position + 1:
            stack.clear()
        while stack and number.rpartition(".")[0] != stack[-1][0]:
            stack.pop()
        if stack:
            parent = stack[-1][1].id
            parents[doc.id] = parent
            children.setdefault(parent, []).append(doc.id)
        if number and _lead(doc.text) and len(list(_CLAUSE.finditer(doc.text))) == 1:
            stack.append((number, doc))
        previous_position = position

    if not parents:
        return documents, parents
    ranks = {doc.id: rank for rank, doc in enumerate(documents)}
    scored = []
    for doc in documents:
        parts = [doc.text]
        if doc.id in parents:
            parts.insert(0, texts[parents[doc.id]])
        if doc.id in children:
            # The fused search already ranks the items by relevance to the
            # question. Include the best one first for bounded model inputs.
            parts.extend(texts[key] for key in sorted(children[doc.id], key=ranks.__getitem__))
        scored.append(RerankDocument(id=doc.id, text="\n".join(parts)))
    return scored, parents
