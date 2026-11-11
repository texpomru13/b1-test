"""Assemble eval/reports/REPORT.md + eval/reports/results.json from measured artifacts.

    python3 eval/report.py --runs balanced-9b,balanced-9b-nothink,low-4b-nothink

Only numbers that exist in eval/work/** are printed; missing artifacts are shown
as `not_run`. Every rate is printed with its numerator/denominator.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"
REPORTS = ROOT / "eval" / "reports"
TARGETS = {
    "citation_span_validity": ("Citation span validity", "100%", 1.0),
    "cross_doc_leakage_citations": ("Cross-document leakage (citations)", "0", 0.0),
    "positive_hit_at_candidates": ("Positive retrieval hit@candidates (60)", ">=95%", 0.95),
    "context_evidence_coverage_all": ("Context evidence coverage (all spans)", ">=90%", 0.90),
    "correct_abstention_negatives": ("Correct abstention on negatives", ">=95%", 0.95),
    "structured_validity_200": ("Structured response validity", "100%", 1.0),
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fmt(r: dict | None) -> str:
    if not r or r.get("den") in (None, 0):
        return "n/a (0 cases)" if r else "not_run"
    return f"{r['num']}/{r['den']} = {100 * r['value']:.1f}%"


def verdict(key: str, r: dict | None) -> str:
    if not r or r.get("value") is None:
        return "not_run"
    target = TARGETS[key][2]
    ok = r["value"] <= target if key == "cross_doc_leakage_citations" else r["value"] >= target
    return "PASS" if ok else "FAIL"


def run_section(name: str) -> list[str]:
    lines = [f"### Run `{name}`", ""]
    meta = load(WORK / "runs" / name / "meta.json")
    if meta is None:
        return lines + ["not_run", ""]
    lines.append(f"Started {meta['started_at']}, API {meta['api']}, git `{meta['git_head'][:12]}`.")
    lines.append("")
    for split in ("all", "holdout", "dev"):
        m = load(WORK / "runs" / name / f"metrics-{split}.json")
        if m is None:
            continue
        lines += [f"**Split `{split}`** — requests {m['n_requests']} (gold {m['n_gold']}: "
                  f"answerable {m['n_answerable']}, no-answer {m['n_no_answer']}; "
                  f"isolation probes {m['n_isolation']})", "",
                  "| Metric | Target | Measured | Verdict |", "|---|---|---|---|"]
        for key, (label, target, _) in TARGETS.items():
            lines.append(f"| {label} | {target} | {fmt(m.get(key))} | {verdict(key, m.get(key))} |")
        lines += [
            f"| Isolation probes answered found=false | 100% | "
            f"{fmt(m.get('isolation_correct_not_found'))} | — |",
            f"| Answerable → found=true | — | {fmt(m.get('answerable_found'))} | — |",
            f"| Answerable → found=false (missed) | — | {fmt(m.get('answerable_not_found'))} | — |",
            f"| Answerable → error (5xx/timeout) | 0 | {fmt(m.get('answerable_errors'))} | — |",
            f"| No-answer → error | 0 | {fmt(m.get('negatives_errors'))} | — |",
            f"| found=true citing a gold span | — | {fmt(m.get('found_cites_gold_span'))} | — |",
            f"| Mean gold-span recall in context | — | {m.get('mean_span_recall_context')} | — |",
            f"| nDCG@10 (reranked) | — | {m.get('ndcg@10_rerank')} | — |",
            "",
        ]
        cats = m.get("by_category", {})
        if cats:
            lines += ["Per category (answerable: found=true; no_answer: found=false):", "",
                      "| Category | Correct outcome |", "|---|---|"]
            lines += [f"| {c} | {fmt(v)} |" for c, v in cats.items()]
            lines.append("")
        lat = m.get("latency_ms_200", {})
        if lat.get("n"):
            lines += [f"Latency of 200 responses (client wall time, sequential): n={lat['n']}, "
                      f"p50={lat['p50'] / 1000:.1f}s, p95={lat['p95'] / 1000:.1f}s, "
                      f"max={lat['max'] / 1000:.1f}s. Stage p50 ms: {m.get('stage_ms_p50')}.", ""]
        j = load(WORK / "runs" / name / f"judge-{split}.json")
        if j:
            lines += [f"LLM-judged (judge `{j['judge']}`, local):", "",
                      "| Metric | Value |", "|---|---|",
                      f"| Expected-fact recall, all answerable | {fmt(j['fact_recall_all_answerable'])} |",
                      f"| Expected-fact recall, found=true only | "
                      f"{fmt(dict(j['fact_recall_found_only'], value=(j['fact_recall_found_only']['num'] / j['fact_recall_found_only']['den']) if j['fact_recall_found_only']['den'] else None))} |",
                      f"| Fully complete answers (all facts) | "
                      f"{j['complete_answers_all_answerable']['num']}/{j['complete_answers_all_answerable']['den']} |",
                      f"| Answers asserting a forbidden fact | "
                      f"{j['answers_with_forbidden_fact']['num']}/{j['answers_with_forbidden_fact']['den']} |",
                      f"| Judge output errors | {j['judge_errors']} |"]
            for k, v in j.get("ragas", {}).items():
                mean = "n/a" if v["mean"] is None else f"{v['mean']:.3f}"
                lines.append(f"| RAGAS {k} | {mean} (n={v['n']}, NaN={v['nan']}) |")
            lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    args = ap.parse_args()
    runs = [r for r in args.runs.split(",") if r]
    gold = [json.loads(x) for x in (ROOT / "eval/gold/gold.jsonl").read_text().splitlines() if x]
    lines = [
        "# Stage-9 evaluation report", "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} from artifacts in "
        "`eval/work/` (see `eval/README.md` for how to reproduce).", "",
        f"Gold set: {len(gold)} cases over {len({c['doc_fixture_id'] for c in gold})} documents "
        f"(dev {sum(c['split'] == 'dev' for c in gold)}, "
        f"holdout {sum(c['split'] == 'holdout' for c in gold)}); agent-drafted, quotes "
        "machine-verified, NOT human-reviewed.", "",
    ]
    for r in runs:
        lines += run_section(r)
    abl = load(WORK / "ablation" / "balanced.metrics.json")
    lines += ["## Retrieval ablation (same gold spans, published balanced index)", ""]
    if abl:
        keys = ["recall@1", "recall@5", "recall@8", "recall@20", "recall@60", "hit@60",
                "all_spans@8", "ndcg@10", "mrr"]
        for split in ("all", "holdout"):
            lines += [f"Split `{split}` (n={abl[split]['dense']['n']} answerable):", "",
                      "| Method | " + " | ".join(keys) + " |",
                      "|---|" + "---|" * len(keys)]
            for meth, vals in abl[split].items():
                lines.append(f"| {meth} | " + " | ".join(str(vals.get(k)) for k in keys) + " |")
            lines.append("")
    else:
        lines += ["not_run", ""]
    comp = load(WORK / "compare" / "report.json")
    lines += ["## Encoder / reranker comparison (fixed chunks, frozen candidates)", ""]
    if comp:
        keys = ["recall@1", "recall@5", "recall@8", "recall@20", "recall@60", "all_spans@8",
                "ndcg@10"]
        lines += ["| Encoder | Mode | " + " | ".join(keys) + " | encode s |",
                  "|---|---|" + "---|" * (len(keys) + 1)]
        for enc, e in comp["encoders"].items():
            for mode in ("dense[all]", "hybrid_rrf60[all]", "dense[holdout]",
                         "hybrid_rrf60[holdout]"):
                lines.append(f"| {enc} | {mode} | " + " | ".join(str(e[mode].get(k))
                                                               for k in keys)
                             + f" | {e['encode_s']} |")
        lines.append("")
        lines += ["| Reranker | Candidates from | " + " | ".join(keys) + " | score s |",
                  "|---|---|" + "---|" * (len(keys) + 1)]
        for rr, e in comp["rerankers"].items():
            for k2, v in e.items():
                if isinstance(v, dict):
                    lines.append(f"| {rr} | {k2} | " + " | ".join(str(v.get(k)) for k in keys)
                                 + f" | {e['score_s']} |")
        lines.append("")
    else:
        lines += ["not_run", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
