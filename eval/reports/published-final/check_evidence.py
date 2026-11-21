"""Exercise four evidence controls without a model, using the measured context and source IDs."""

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


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path, help="unpacked inputs.tar.gz")
    args = parser.parse_args()
    report = Path(__file__).resolve().parent
    labels = {r["case_id"]: r for r in rows(args.work / "gold.jsonl")}
    records = {r["req_id"]: r for r in rows(report / "responses.jsonl")}
    controls = {
        "bulletin-07": [
            ("source-1.1", "Для срока от 91 до 180 дней значение составляет 15,15%."),
            ("source-1.1", "Для срока свыше 1 года значение составляет 15,05%."),
        ],
        "nairobi-01": [("source-1.1", labels["nairobi-01"]["evidence"][0]["text"])],
        "echr-05": [("source-1.1", labels["echr-05"]["evidence"][0]["text"]),
                    ("source-2.1", labels["echr-05"]["evidence"][1]["text"])],
        "space-02": [],
    }
    for case_id, selected in controls.items():
        result = records[case_id]
        label = labels[case_id]
        export = json.loads((args.work / "index" / f"{result['target_fixture']}.json").read_text())
        by_id = {item["chunk_id"]: item for item in export["chunks"]}
        diag = result["body"]["diagnostics"]
        sources = {
            f"source-{i}": ContextChunk(**by_id[key], rerank_score=diag["rerank_scores"][key])
            for i, key in enumerate(diag["context_chunk_ids"], 1)
        }
        spans = issue_evidence(sources)
        present = [any(e["text"] in chunk.text for chunk in sources.values())
                   for e in label["evidence"]]
        print(case_id, "полные эталонные выдержки в контексте", present)
        for key, text in selected:
            claim = Claim(text=" ".join(text.split()), evidence=[key])
            print(" ", key, spans[key].char_start, spans[key].char_end, claim.text)
            try:
                located = validate_claims([claim], sources, question=result["question"],
                                          spans=spans, render_conditional_rules=True)
                qualify_numeric_claims([claim], located)
            except ProvenanceError as exc:
                print("  отклонено:", str(exc)[:170])
            else:
                print("  принято")


if __name__ == "__main__":
    main()
