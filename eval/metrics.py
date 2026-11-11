"""Deterministic (non-LLM) metrics for one QA run (plan §11.3).

    uv run python eval/metrics.py --run balanced-9b [--split holdout]

Reads eval/gold/gold.jsonl, eval/work/index/*.json, eval/work/runs/<run>/responses.jsonl
and writes eval/work/runs/<run>/metrics.json (+ per-case rows in cases.jsonl).

Span coverage: a gold quote occurrence [s, e) is covered by a set of chunks when
>= 80% of its characters fall inside the union of the chunks' [char_start, char_end).
A gold span counts as covered when ANY of its occurrences is covered.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.qa import AnswerResponse
from spans import chunk_relevant, ndcg, span_covered
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def page_of(pages: list[dict], offset: int) -> int:
    for p in pages:
        if p["char_start"] <= offset < p["char_end"]:
            return int(p["page"])
    return int(pages[-1]["page"])


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))
    return vals[idx]


def ratio(num: int, den: int) -> dict[str, Any]:
    return {"num": num, "den": den, "value": round(num / den, 4) if den else None}


def outcome(rec: dict) -> str:
    st = rec.get("http_status")
    if st == 200:
        return "found" if rec["body"].get("found") else "not_found"
    if st is None:
        return "client_error"
    return f"http_{st}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    args = ap.parse_args()

    gold = {c["case_id"]: c for c in load_jsonl(ROOT / "eval" / "gold" / "gold.jsonl")}
    index = {p.stem: json.loads(p.read_text(encoding="utf-8"))
             for p in (WORK / "index").glob("*.json")}
    by_doc_id = {v["doc_id"]: v for v in index.values()}
    chunk_rng = {fid: {c["chunk_id"]: (c["char_start"], c["char_end"]) for c in v["chunks"]}
                 for fid, v in index.items()}
    run_dir = WORK / "runs" / args.run
    records = load_jsonl(run_dir / "responses.jsonl")
    if args.split != "all":
        records = [r for r in records if gold[r["case_id"]]["split"] == args.split]

    rows = []
    agg: dict[str, Any] = {"outcomes": {}, "by_category": {}}
    schema_ok = schema_total = 0
    cit_ok = cit_total = 0
    leak = leak_checked = 0
    walls: list[float] = []
    stage_vals: dict[str, list[float]] = {}
    for rec in records:
        case = gold[rec["case_id"]]
        oc = outcome(rec)
        body = rec.get("body") or {}
        row: dict[str, Any] = {"req_id": rec["req_id"], "kind": rec["kind"],
                               "case_id": case["case_id"], "category": case["category"],
                               "answerable": case["answerable"], "outcome": oc,
                               "wall_ms": rec.get("wall_ms")}
        target = by_doc_id[rec["doc_id"]]
        tfid = target["fixture_id"]
        if rec.get("http_status") == 200:
            schema_total += 1
            try:
                AnswerResponse.model_validate(body)
                schema_ok += 1
                row["schema_valid"] = True
            except ValidationError as exc:
                row["schema_valid"] = False
                row["schema_error"] = str(exc)[:300]
            text, pages = target["canonical_text"], target["pages"]
            bad = []
            for c in body.get("citations", []):
                cit_total += 1
                leak_checked += 1
                rng = chunk_rng[tfid].get(c["chunk_id"])
                if rng is None:
                    leak += 1
                    bad.append(f"chunk {c['chunk_id']} not in queried doc run")
                    continue
                ok = (text[c["char_start"]:c["char_end"]] == c["text"]
                      and rng[0] <= c["char_start"] and c["char_end"] <= rng[1]
                      and c["page"] == page_of(pages, c["char_start"])
                      and c["page_to"] == page_of(pages, c["char_end"] - 1))
                cit_ok += ok
                if not ok:
                    bad.append(f"citation {c['id']} span/page mismatch")
            row["citation_problems"] = bad
            walls.append(rec["wall_ms"])
            diag = body.get("diagnostics") or {}
            for k, v in (diag.get("stage_ms") or {}).items():
                stage_vals.setdefault(k, []).append(v)
        agg["outcomes"].setdefault(rec["kind"], {}).setdefault(oc, 0)
        agg["outcomes"][rec["kind"]][oc] += 1

        if rec["kind"] == "gold" and case["answerable"] and rec.get("http_status") == 200:
            diag = body.get("diagnostics") or {}
            cands = list((diag.get("rerank_scores") or {}).keys())  # reranked order
            ctx = diag.get("context_chunk_ids") or []
            rng = chunk_rng[tfid]
            spans = case["gold_spans"]
            cand_r = [rng[c] for c in cands if c in rng]
            ctx_r = [rng[c] for c in ctx if c in rng]
            row["hit_at_cands"] = any(span_covered(s, cand_r) for s in spans)
            row["all_spans_in_cands"] = all(span_covered(s, cand_r) for s in spans)
            row["ctx_any"] = any(span_covered(s, ctx_r) for s in spans)
            row["ctx_all"] = all(span_covered(s, ctx_r) for s in spans)
            row["span_recall_ctx"] = sum(span_covered(s, ctx_r) for s in spans) / len(spans)
            row["ndcg10_rerank"] = ndcg(cand_r, spans, 10)
            for k in (1, 3, 5, 8, 20, 60):
                row[f"recall@{k}"] = sum(span_covered(s, cand_r[:k]) for s in spans) / len(spans)
            cited = [(c["char_start"], c["char_end"]) for c in body.get("citations", [])]
            row["cited_gold_overlap"] = (
                any(chunk_relevant(c, spans) for c in cited) if cited else None
            )
            row["n_candidates"] = len(cands)
            row["n_context"] = len(ctx)
        rows.append(row)

    gold_rows = [r for r in rows if r["kind"] == "gold"]
    pos = [r for r in gold_rows if r["answerable"]]
    neg = [r for r in gold_rows if not r["answerable"]]
    pos200 = [r for r in pos if "hit_at_cands" in r]
    iso = [r for r in rows if r["kind"] == "isolation"]

    def mean(key: str, rs: list[dict]) -> float | None:
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    agg.update({
        "run": args.run, "split": args.split,
        "n_requests": len(rows), "n_gold": len(gold_rows),
        "n_answerable": len(pos), "n_no_answer": len(neg), "n_isolation": len(iso),
        "structured_validity_200": ratio(schema_ok, schema_total),
        "citation_span_validity": ratio(cit_ok, cit_total),
        "cross_doc_leakage_citations": ratio(leak, leak_checked),
        "isolation_correct_not_found": ratio(
            sum(r["outcome"] == "not_found" for r in iso), len(iso)),
        "isolation_found_true": [r["req_id"] for r in iso if r["outcome"] == "found"],
        "positive_hit_at_candidates": ratio(sum(r["hit_at_cands"] for r in pos200), len(pos)),
        "all_spans_in_candidates": ratio(sum(r["all_spans_in_cands"] for r in pos200), len(pos)),
        "context_evidence_coverage_all": ratio(sum(r["ctx_all"] for r in pos200), len(pos)),
        "context_evidence_coverage_any": ratio(sum(r["ctx_any"] for r in pos200), len(pos)),
        "mean_span_recall_context": mean("span_recall_ctx", pos200),
        "ndcg@10_rerank": mean("ndcg10_rerank", pos200),
        **{f"span_recall@{k}_rerank": mean(f"recall@{k}", pos200) for k in (1, 3, 5, 8, 20, 60)},
        "answerable_found": ratio(sum(r["outcome"] == "found" for r in pos), len(pos)),
        "answerable_not_found": ratio(sum(r["outcome"] == "not_found" for r in pos), len(pos)),
        "answerable_errors": ratio(
            sum(r["outcome"] not in ("found", "not_found") for r in pos), len(pos)),
        "found_cites_gold_span": ratio(
            sum(bool(r.get("cited_gold_overlap")) for r in pos if r["outcome"] == "found"),
            sum(r["outcome"] == "found" for r in pos)),
        "correct_abstention_negatives": ratio(
            sum(r["outcome"] == "not_found" for r in neg), len(neg)),
        "negatives_errors": ratio(
            sum(r["outcome"] not in ("found", "not_found") for r in neg), len(neg)),
        "latency_ms_200": {
            "n": len(walls), "p50": pct(walls, 0.5), "p95": pct(walls, 0.95),
            "max": max(walls) if walls else None,
        },
        "stage_ms_p50": {k: pct(v, 0.5) for k, v in sorted(stage_vals.items())},
        "stage_ms_p95": {k: pct(v, 0.95) for k, v in sorted(stage_vals.items())},
    })
    for cat in sorted({r["category"] for r in gold_rows}):
        rs = [r for r in gold_rows if r["category"] == cat]
        good = sum((r["outcome"] == "found") if r["answerable"] else
                   (r["outcome"] == "not_found") for r in rs)
        agg["by_category"][cat] = ratio(good, len(rs))
    (run_dir / f"metrics-{args.split}.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1))
    with (run_dir / f"cases-{args.split}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
