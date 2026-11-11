"""Shared span/chunk matching helpers (stdlib only; used by metrics and comparisons).

A gold quote occurrence [s, e) is covered by a set of chunks when >= 80% of its
characters fall inside the union of the chunks' [char_start, char_end). A chunk is
relevant when it overlaps a gold occurrence by >= min(30, len(occurrence)) chars.
"""

from __future__ import annotations

import math

COVER_RATIO = 0.8


def covered(occ: dict, ranges: list[tuple[int, int]]) -> bool:
    s, e = occ["char_start"], occ["char_end"]
    hit = [0] * (e - s)
    for cs, ce in ranges:
        for i in range(max(s, cs), min(e, ce)):
            hit[i - s] = 1
    return sum(hit) >= COVER_RATIO * (e - s)


def span_covered(span: dict, ranges: list[tuple[int, int]]) -> bool:
    return any(covered(o, ranges) for o in span["occurrences"])


def chunk_relevant(rng: tuple[int, int], spans: list[dict]) -> bool:
    cs, ce = rng
    for span in spans:
        for o in span["occurrences"]:
            overlap = min(ce, o["char_end"]) - max(cs, o["char_start"])
            if overlap >= min(30, o["char_end"] - o["char_start"]):
                return True
    return False


def ndcg(ranked: list[tuple[int, int]], spans: list[dict], k: int,
         doc_ranges: list[tuple[int, int]]) -> float:
    """Binary-relevance nDCG@k; the ideal ranking places ALL relevant chunks of the
    document first (relevance = chunk_relevant), so the score is not capped by the
    number of gold spans."""
    gains = [1.0 if chunk_relevant(r, spans) else 0.0 for r in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_rel = min(k, sum(1 for r in doc_ranges if chunk_relevant(r, spans)))
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg else 0.0
