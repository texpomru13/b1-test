"""Limit explicitly named legal articles and protocols to their printed boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.search import ContextChunk

_NUMBER = r"\d+(?:\.\d+)*"
_ARTICLE_NO = rf"(?:{_NUMBER}|[IVXLCDM]+)"
_REQUEST = re.compile(rf"\b(?:(?P<article>стать[яиюеёй]+|ст\.)|"
                      rf"(?P<protocol>протокол(?:а|е|у|ом)?))\s*(?:№\s*)?({_ARTICLE_NO})\b", re.I)
_MORE = re.compile(r"\s*(?:[-–—,]|и\s)\s*\d")
_RELATIVE = re.compile(r",\s*котор\w*\b[^,;?!]*$", re.I)
_GROUP_REF = re.compile(r"\b(?:упомянут|перечисленн|названн|указанн)\w*\s+в\s*$", re.I)
_HEADING = re.compile(rf"^[ \t]*(?:(?P<article>статья)\s+(?P<a>{_ARTICLE_NO})"
                      rf"|(?P<protocol>протокол)\s*(?:№\s*)?(?P<p>{_NUMBER})"
                      r"|(?P<boundary>протокол|раздел\s+[IVXLCDM]+))[ \t]*$", re.I | re.M)


@dataclass(frozen=True)
class Heading:
    chunk: ContextChunk
    start: int
    end: int
    number: str

    @property
    def span(self) -> tuple[int, int]:
        return self.chunk.char_start + self.start, self.chunk.char_start + self.end


def _slice(chunk: ContextChunk, left: int, right: int) -> ContextChunk:
    text = chunk.text[left:right]
    start = chunk.char_start + left + len(text) - len(text.lstrip())
    end = chunk.char_start + right - (len(text) - len(text.rstrip()))
    segments = [s for s in chunk.segments if s["char_start"] < end and start < s["char_end"]]
    return replace(chunk, text=text.strip(), char_start=start, char_end=end, segments=segments,
                   page_from=segments[0]["page"] if segments else chunk.page_from,
                   page_to=segments[-1]["page"] if segments else chunk.page_to)


def _named(question: str) -> dict[str, str]:
    request: dict[str, str] = {}
    for ref in _REQUEST.finditer(question):
        kind = "article" if ref["article"] else "protocol"
        # A party described by a cross-reference may exercise its right under
        # an entirely different provision. Keep the full scoped candidate set.
        before = question[:ref.start()]
        if (_GROUP_REF.search(before)
                or (_RELATIVE.search(before) and "," in question[ref.end():])):
            continue
        if (kind in request and request[kind] != ref[3]) or _MORE.match(question, ref.end()):
            return {}
        request[kind] = ref[3].upper()
    return request


def article_gaps(question: str, units: dict[str, list[ContextChunk]]) -> set[int]:
    request = _named(question)
    if not request:
        return set()
    chunks = [u[-1] for u in units.values()]
    positions = {c.chunk_index for c in chunks}
    anchor_kind = "protocol" if "protocol" in request else "article"
    headings = sorted((c.chunk_index, m["p"] if anchor_kind == "protocol" else m["a"])
                      for c in chunks for m in _HEADING.finditer(c.text) if m[anchor_kind])
    gaps = set()
    for i, (start, value) in enumerate(headings):
        if value.upper() != request[anchor_kind]:
            continue
        stop = min(start + 48, headings[i + 1][0] if i + 1 < len(headings) else start + 48)
        tail = max(pos for pos in positions if start <= pos <= stop)
        gaps.update(set(range(start + 1, tail)) - positions)
    return gaps


def article_units(question: str, units: dict[str, list[ContextChunk]]) -> set[str] | None:
    """Return eligible fused IDs only when the requested heading exists in the PDF.

    A gap clears both ancestors. An intervening protocol resets its articles,
    even when an article with the same number occurs in both instruments.
    """
    request = _named(question)
    if not request:
        return None
    spans: dict[str, list[tuple[int, int, tuple[Heading, ...]]]] = {}
    active: dict[str, Heading] = {}
    seen: dict[str, set[str]] = {"article": set(), "protocol": set()}
    previous = -2
    for key, unit in sorted(units.items(), key=lambda pair: pair[1][-1].chunk_index):
        chunk = unit[-1]
        if chunk.chunk_index != previous + 1:
            active.clear()
        previous = chunk.chunk_index
        start = 0
        matches = list(_HEADING.finditer(chunk.text))
        for m in [*matches, None]:
            stop = m.start() if m else len(chunk.text)
            if (chunk.text[start:stop].strip() and
                    all(k in active and active[k].number == value for k, value in request.items())):
                chain = tuple(active[k] for k in ("protocol", "article") if k in request)
                regions = spans.setdefault(key, [])
                if regions and regions[-1][2] == chain:
                    regions[-1] = (regions[-1][0], stop, chain)
                else:
                    regions.append((start, stop, chain))
            if m is None:
                break
            if m["boundary"]:
                active.pop("article", None)
                if m["boundary"].casefold() == "протокол":
                    active.clear()
            else:
                kind = "article" if m["article"] else "protocol"
                if kind == "protocol":
                    active.clear()
                number = (m["a"] or m["p"]).upper()
                seen[kind].add(number)
                active[kind] = Heading(chunk, m.start(), m.end(), number)
            start = m.start()
    if any(value not in seen[k] for k, value in request.items()):
        return None
    # If two indexed instruments both have this article and the user did not
    # specify one, let the semantic verifier disambiguate; do not choose by order.
    anchors = {chain[-1].span for regions in spans.values() for _, _, chain in regions}
    roman_article = set(request) == {"article"} and request["article"].isalpha()
    if len(anchors) > 1 and not roman_article:
        return None

    # Two adjacent titles can share one Qdrant source. A contiguous citation
    # contains both only if there is no intervening instrument or article.
    for key, regions in spans.items():
        for pos, (left, right, chain) in enumerate(regions):
            if len(chain) < 2 or chain[0].chunk.chunk_id != chain[1].chunk.chunk_id:
                continue
            protocol, article = chain
            between = protocol.chunk.text[protocol.end:article.start]
            if _HEADING.search(between) or len(between) > 350:
                return None
            joined = Heading(protocol.chunk, protocol.start, article.end, article.number)
            if key == article.chunk.chunk_id and left == article.start:
                left = protocol.start
            regions[pos] = left, right, (joined,)
    bounded = {chain[-1].span for key, regions in spans.items() for _, end, chain in regions
               if _HEADING.match(units[key][-1].text, end)}
    eligible = set()
    views = {key: _slice(units[key][-1], regions[0][0], regions[0][1])
             for key, regions in spans.items() if len(regions) == 1}
    for key, view in views.items():
        heading_chain = spans[key][0][2]
        label = " · ".join(h.chunk.text[h.start:h.end].strip() for h in heading_chain)
        parents: list[ContextChunk] = []
        parent = None
        for h in heading_chain:
            if h.chunk.chunk_id == key or (parent and parent.chunk.chunk_id == h.chunk.chunk_id):
                continue
            header = views.get(h.chunk.chunk_id) or _slice(h.chunk, h.start, h.end)
            parents.append(replace(
                header, list_heading_id=parent.chunk.chunk_id if parent else None,
                scope_heading_span=parent.span if parent else None,
                scope_label=header.scope_label or label,
            ))
            parent = h
        # Keep an existing governing list intact if all of its sources also belong
        # to this occurrence. Never attach an outside introductory clause.
        ancestors = units[key][:-1]
        if any(c.chunk_id not in views for c in ancestors):
            continue
        for ancestor in ancestors:
            if ancestor.chunk_id not in {c.chunk_id for c in parents}:
                nested = views[ancestor.chunk_id]
                parents.append(replace(
                    nested, list_heading_id=parents[-1].chunk_id if parents else None,
                    scope_label=nested.scope_label or label,
                ))
        governing = parent or heading_chain[-1]
        span = None if ancestors else governing.span
        parents.append(replace(view, list_heading_id=parents[-1].chunk_id if parents else None,
                             scope_heading_span=span, scope_label=view.scope_label or label))
        units[key] = parents
        eligible.add(key)
    if roman_article:
        # A collection of treaties can repeat e.g. "Article III". Keep each
        # short occurrence together across a page break, with original IDs.
        # The reranker still has to find the applicable instrument.
        for anchor in bounded:
            members = [key for key in views if key in eligible
                       and spans[key][0][2][-1].span == anchor]
            if 1 < len(members) <= 4 and sum(len(views[key].text) for key in members) <= 1400:
                parts = {c.chunk_id: c for key in members for c in units[key]}
                for key in members:
                    units[key] = [*(c for k, c in parts.items() if k != key), parts[key]]
    return None if roman_article else eligible
