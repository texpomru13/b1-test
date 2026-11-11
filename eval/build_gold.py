"""Validate gold drafts against the canonical text and build eval/gold/gold.jsonl.

Run (stdlib only):  python3 eval/build_gold.py

Inputs:
  eval/gold/drafts/<fixture>.jsonl  — drafted cases (question, quotes, facts, ...)
  eval/work/index/<fixture>.json    — canonical text + page map (export_index)
Checks (any failure -> the case is rejected and listed; exit code 1):
  - every gold quote is an EXACT substring of the canonical text (no fuzzy match);
  - answerable cases have >=1 quote and >=1 expected fact; no-answer cases have none;
  - category / split values are known; questions are unique per document.
Offsets and pages are recovered by the script (never trusted from the draft); a
quote occurring several times keeps ALL occurrences (any of them counts as hit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / "eval" / "gold" / "drafts"
INDEX = ROOT / "eval" / "work" / "index"
OUT = ROOT / "eval" / "gold" / "gold.jsonl"
CATEGORIES = {"exact", "paraphrase", "multi", "negation", "no_answer"}
# Split by DOCUMENT (plan §11.2): dev documents are never used for reporting
# headline numbers, holdout documents are never used for tuning.
DEV_FIXTURES = {"01", "05", "13", "15"}


def page_of(pages: list[dict], offset: int) -> int:
    for span in pages:
        if span["char_start"] <= offset < span["char_end"]:
            return int(span["page"])
    return int(pages[-1]["page"])


def occurrences(text: str, quote: str) -> list[int]:
    found, start = [], text.find(quote)
    while start != -1:
        found.append(start)
        start = text.find(quote, start + 1)
    return found


def main() -> int:
    cases, rejected = [], []
    for draft in sorted(DRAFTS.glob("*.jsonl")):
        fid = draft.stem
        index = json.loads((INDEX / f"{fid}.json").read_text(encoding="utf-8"))
        text, pages = index["canonical_text"], index["pages"]
        seen_questions: set[str] = set()
        for n, line in enumerate(draft.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            problems = []
            question = raw.get("question", "").strip()
            category = raw.get("category")
            answerable = bool(raw.get("answerable"))
            quotes = [q for q in raw.get("gold_quotes", []) if q]
            facts = raw.get("expected_facts", [])
            if category not in CATEGORIES:
                problems.append(f"unknown category {category!r}")
            if (category == "no_answer") == answerable:
                problems.append("answerable flag contradicts category")
            if answerable and (not quotes or not facts):
                problems.append("answerable case needs gold_quotes and expected_facts")
            if not answerable and quotes:
                problems.append("no-answer case must not carry gold quotes")
            if question.lower() in seen_questions:
                problems.append("duplicate question")
            seen_questions.add(question.lower())
            spans = []
            for quote in quotes:
                starts = occurrences(text, quote)
                if not starts:
                    problems.append(f"quote not in canonical text: {quote[:60]!r}")
                    continue
                spans.append({
                    "text": quote,
                    "occurrences": [
                        {"char_start": s, "char_end": s + len(quote),
                         "page": page_of(pages, s)} for s in starts
                    ],
                })
            case_id = f"{fid}-{n:02d}"
            if problems:
                rejected.append((case_id, question, problems))
                continue
            first = spans[0]["occurrences"][0] if spans else None
            cases.append({
                "case_id": case_id,
                "doc_fixture_id": fid,
                "question": question,
                "answerable": answerable,
                "category": category,
                "reference_answer": raw.get("reference_answer", ""),
                "expected_facts": facts,
                "forbidden_facts": raw.get("forbidden_facts", []),
                "gold_spans": spans,
                "page": first["page"] if first else None,
                "section": raw.get("section"),
                "entity_specific": bool(raw.get("entity_specific", False)),
                "split": "dev" if fid in DEV_FIXTURES else "holdout",
                "provenance": "agent-drafted; quotes machine-verified as exact "
                              "substrings; NOT human-reviewed",
                "notes": raw.get("notes", ""),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for case in cases:
        by_cat[case["category"]] = by_cat.get(case["category"], 0) + 1
    splits = {s: sum(c["split"] == s for c in cases) for s in ("dev", "holdout")}
    print(f"accepted {len(cases)} cases -> {OUT.relative_to(ROOT)}; by category {by_cat}; "
          f"splits {splits}")
    for case_id, question, problems in rejected:
        print(f"REJECTED {case_id}: {question[:70]!r}: {'; '.join(problems)}")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
