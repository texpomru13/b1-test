"""Validate rate bands and numbered treaty clauses without calling a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.qa.evidence import issue_evidence  # noqa: E402
from app.qa.grounding import ProvenanceError, qualify_numeric_claims, validate_claims  # noqa: E402
from app.qa.schemas import Claim  # noqa: E402
from app.retrieval.search import ContextChunk  # noqa: E402


def records(path: Path, key: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {r[key]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="unpacked inputs.tar.gz")
    parser.add_argument("--responses", type=Path,
                        default=Path(__file__).with_name("responses.jsonl"))
    args = parser.parse_args()
    labels = records(args.work / "gold.jsonl", "case_id")
    results = records(args.responses, "req_id")
    treaty_clause = labels["nairobi-01"]["evidence"][0]["text"]
    controls = {
        "bulletin-07": [
            ("Для срока от 91 до 180 дней значение составляет 15,15%.", True),
            ("Для срока свыше 1 года значение составляет 15,05%.", True),
            ("Для срока свыше 1 года значение составляет 15,85%.", False),
        ],
        "nairobi-01": [(" ".join(treaty_clause.split()), True)],
    }
    for case_id, examples in controls.items():
        row = results[case_id]
        export = json.loads((args.work / "index" / f"{row['target_fixture']}.json").read_text())
        if row["doc_id"] != export["doc_id"]:
            raise ValueError(f"{case_id}: request belongs to another index")
        by_id = {c["chunk_id"]: c for c in export["chunks"]}
        diag = row["body"]["diagnostics"]
        context = {
            f"source-{i}": ContextChunk(**by_id[key], rerank_score=diag["rerank_scores"][key])
            for i, key in enumerate(diag["context_chunk_ids"], 1)
        }
        spans = issue_evidence(context)
        anchor = labels[case_id]["evidence"][0]["text"]
        key, quote = next((k, s) for k, s in spans.items() if anchor in s.quote_raw)
        raw = export["canonical_text"][quote.char_start:quote.char_end]
        page = export["pages"][quote.page - 1]
        if raw != quote.quote_raw or not page["char_start"] <= quote.char_start < page["char_end"]:
            raise ValueError(f"{case_id}: source coordinates do not match the PDF export")
        if case_id == "nairobi-01" and raw.strip() != treaty_clause.strip():
            raise ValueError("requested numbered clause was merged with another clause")
        for text, expected in examples:
            claim = Claim(text=text, evidence=[key])
            try:
                located = validate_claims([claim], context, question=row["question"], spans=spans,
                                          render_conditional_rules=True)
                qualify_numeric_claims([claim], located)
                accepted = True
            except ProvenanceError:
                accepted = False
            if accepted != expected:
                raise ValueError(f"{case_id}: unexpected decision for {text}")
            print(case_id, key, f"p{quote.page} [{quote.char_start}:{quote.char_end}]",
                  "принято" if accepted else "отклонено", text)


if __name__ == "__main__":
    main()
