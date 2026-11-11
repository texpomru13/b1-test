"""Symmetric lexical preprocessing for BM25 (LEXICAL_NORMALIZER_VERSION=1).

Ported from the stage-0 S3 prototype (roundtrip-verified against the real
fastembed pipeline). Applied IDENTICALLY to documents and queries, for the BM25
input only — citation text and canonical offsets are never altered.

Rules v1:
- ё -> е on the whole BM25 input (fastembed does not normalize it; S3 evidence).
- Canonical synthetic tokens appended for exact legal designations that the
  default tokenizer/stemmer would otherwise destroy:
    clause numbers  4.2.1      -> sec4x2x1
    percents        0,1%       -> pct0x1
    references      № АБ-123/24 -> refабx123x24
    spaced amounts  1 000 000  -> amt1000000
  Tokens use only [a-z0-9а-яе] + 'x' so the tokenizer keeps them whole and the
  Russian Snowball stemmer cannot mangle them.
"""

from __future__ import annotations

import re

LEXICAL_NORMALIZER_ID = "lexnorm-v1:yo2e+sec/pct/ref/amt"

_RE_CLAUSE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)+)(?![\d.])")
_RE_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_RE_REF = re.compile(r"№\s*([0-9A-Za-zА-Яа-яЁё]+(?:[-/][0-9A-Za-zА-Яа-яЁё]+)*)")
_RE_AMT = re.compile(r"(?<!\d)(\d{1,3}(?:[   ]\d{3})+)(?!\d)")


def synthetic_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _RE_CLAUSE.finditer(text):
        tokens.append("sec" + match.group(1).replace(".", "x"))
    for match in _RE_PCT.finditer(text):
        tokens.append("pct" + match.group(1).replace(",", "x").replace(".", "x"))
    for match in _RE_REF.finditer(text):
        norm = match.group(1).lower().replace("ё", "е")
        tokens.append("ref" + re.sub(r"[-/]", "x", norm))
    for match in _RE_AMT.finditer(text):
        tokens.append("amt" + re.sub(r"[   ]", "", match.group(1)))
    return tokens


def prepare_bm25_input(text: str) -> str:
    """ё->е plus appended synthetic tokens; for BM25 encoding only."""
    normalized = text.replace("ё", "е").replace("Ё", "Е")
    tokens = synthetic_tokens(normalized)
    return (normalized + " " + " ".join(tokens)).strip() if tokens else normalized
