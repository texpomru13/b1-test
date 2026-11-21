"""Preserve the two separate bases of an overall percentage and its scoped sublimit."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.qa.grounding import LocatedQuote

_PERCENT = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s*(?:%|процент[а-яё]*)", re.I)
_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+|%", re.UNICODE)
_PAREN = re.compile(r"\(([^()]{1,650})\)")
_CLOSED_EARLY = re.compile(r"\((?P<scope>[^()\n,:;]{2,110})\)\s*(?P<dash>[–—-])\s*")


def _tokens(text: str) -> list[str]:
    return [t.casefold().replace("ё", "е") for t in _TOKEN.findall(text)]


def _first_rate_and_base(prefix: str) -> list[str]:
    rates = list(_PERCENT.finditer(prefix))
    if not rates:
        return []
    tail = prefix[rates[-1].start():]
    # Do not borrow a percentage or its basis from a separate sentence.
    if len(tail) > 140 or re.search(r"[;!?]|(?<=[а-яё])\.\s+[А-ЯЁ]", tail):
        return []
    value_and_base = _tokens(tail)
    return value_and_base if len(value_and_base) >= 4 else []


def _scope_before_dash(left: str, scope: str, q: LocatedQuote) -> bool:
    match = re.match(r"\s*" + r"\s+".join(map(re.escape, scope.split())), left, re.I)
    if not match:
        return False
    rest = left[match.end():]
    if not rest.strip():
        return True
    # At a physical page break an otherwise continuous parenthesis can contain
    # a running title and a folio. Only ignore this narrow PDF layout pattern.
    footer = re.fullmatch(r"\s*\n[^\n]{1,150}\n\s*(\d{1,4})\s*\n\s*", rest)
    return bool(footer and q.page < q.page_to and q.page <= int(footer[1]) <= q.page_to)


def restore_sublimit_punctuation(text: str, quotes: Sequence[LocatedQuote]) -> str:
    for draft in reversed(list(_CLOSED_EARLY.finditer(text))):
        leading = _first_rate_and_base(text[:draft.start()])
        if not leading:
            continue
        tail = text[draft.end():]
        answer_tokens = list(_TOKEN.finditer(tail))
        tail_values = _tokens(tail)
        matches: set[tuple[int, str]] = set()
        relevant = False
        for q in quotes:
            for paren in _PAREN.finditer(q.quote_raw):
                before = _first_rate_and_base(q.quote_raw[:paren.start()])
                if not before:
                    continue
                for dash in re.finditer("[–—-]", paren[1]):
                    if not _scope_before_dash(paren[1][:dash.start()], draft["scope"], q):
                        continue
                    second = paren[1][dash.end():].strip()
                    if len(list(_PERCENT.finditer(second))) != 1:
                        continue
                    second_terms = _tokens(second)
                    if len(second_terms) < 4:
                        continue
                    relevant = True
                    if before == leading and tail_values[:len(second_terms)] == second_terms:
                        matches.add((answer_tokens[len(second_terms) - 1].end(), dash[0]))
        if len(matches) == 1:
            close, original_dash = matches.pop()
            text = (text[:draft.start()] + "(" + draft["scope"] + " " + original_dash + " "
                    + tail[:close] + ")" + tail[close:])
        elif relevant:
            raise ValueError("Уточни, какая база относится к общему пределу и какая — "
                             "к пределу в скобках. Перенеси обе базы и ставки по источнику.")
    return text
