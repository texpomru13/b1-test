"""Retrieval ablation: dense-only / BM25-only / hybrid RRF / hybrid DBSF / hybrid+rerank.

    python3 eval/ablation.py --project b1-eval --name balanced    # runs the queries
    python3 eval/ablation.py --name balanced --report-only        # metrics only

The queries run inside <project>-api-1 (eval/incontainer/ablation.py) against the
PUBLISHED index with the production scope filter; metrics are computed here on
the same gold spans for every method (answerable gold cases only).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from spans import ndcg, span_covered

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"
METHODS = ["dense", "sparse", "hybrid_rrf", "hybrid_dbsf", "rrf_rerank"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def evaluate(ranked: dict[str, dict], gold: list[dict], index: dict[str, dict],
             methods: list[str], ctx_k: int = 8) -> dict:
    out: dict[str, dict] = {}
    for m in methods:
        rows = []
        for case in gold:
            res = ranked.get(case["case_id"])
            if res is None or m not in res:
                continue
            rng = {c["chunk_id"]: (c["char_start"], c["char_end"])
                   for c in index[case["doc_fixture_id"]]["chunks"]}
            r = [rng[c] for c in res[m] if c in rng]
            spans = case["gold_spans"]
            row = {f"recall@{k}": sum(span_covered(s, r[:k]) for s in spans) / len(spans)
                   for k in (1, 3, 5, ctx_k, 20, 60)}
            row["hit@60"] = float(any(span_covered(s, r[:60]) for s in spans))
            row[f"all_spans@{ctx_k}"] = float(all(span_covered(s, r[:ctx_k]) for s in spans))
            row["ndcg@10"] = ndcg(r, spans, 10)
            first = next((i for i in range(len(r)) if any(span_covered(s, r[i:i + 1])
                                                          for s in spans)), None)
            row["mrr"] = 1 / (first + 1) if first is not None else 0.0
            rows.append(row)
        out[m] = {"n": len(rows), **{k: round(statistics.fmean(x[k] for x in rows), 4)
                                     for k in rows[0]}} if rows else {"n": 0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="b1-eval")
    ap.add_argument("--name", default="balanced")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    gold = [c for c in load_jsonl(ROOT / "eval" / "gold" / "gold.jsonl") if c["answerable"]]
    index = {p.stem: json.loads(p.read_text(encoding="utf-8"))
             for p in (WORK / "index").glob("*.json")}
    out_dir = WORK / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{args.name}.raw.json"
    if not args.report_only:
        queries = [{"req_id": c["case_id"], "doc_id": index[c["doc_fixture_id"]]["doc_id"],
                    "question": c["question"]} for c in gold]
        script = (ROOT / "eval" / "incontainer" / "ablation.py").read_text()
        res = subprocess.run(
            ["docker", "exec", "-i", "-e", f"EVAL_QUERIES_JSON={json.dumps(queries)}",
             f"{args.project}-api-1", "python", "-"],
            input=script, capture_output=True, text=True, check=True)
        raw_path.write_text(res.stdout, encoding="utf-8")
    ranked = json.loads(raw_path.read_text(encoding="utf-8"))
    report = {"n_cases": len(gold), "all": evaluate(ranked, gold, index, METHODS)}
    for split in ("dev", "holdout"):
        report[split] = evaluate(ranked, [c for c in gold if c["split"] == split], index,
                                 METHODS)
    (out_dir / f"{args.name}.metrics.json").write_text(json.dumps(report, indent=1))
    for m in METHODS:
        print(m, report["all"][m])


if __name__ == "__main__":
    main()
