"""Measure whether previously complete balanced answers stay complete on a new build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import ROOT, WORK
from provenance import sha256
from qualification_review import blank_review, check_review, model_review, segments, write_json
from run_qa import load_jsonl

REPORTS = ROOT / "eval" / "reports"
BALANCED_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"

CLAIM_REVIEWS = {
    "contracts": ("claim-review.json", "qualification.json"),
    "offers": ("crosscheck-claims.json", "crosscheck.json"),
    "published-final": ("claim-review.json", "qualification.json"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def previous_reviews() -> list[tuple[Path, Path, dict]]:
    accepted = []
    for series, (review_name, score_name) in CLAIM_REVIEWS.items():
        folder = REPORTS / series
        review, score = folder / review_name, load(folder / score_name)
        if (score["review_sha256"] != sha256(review)
                or score["responses_sha256"] != sha256(folder / "responses.jsonl")):
            raise ValueError(f"review changed: {review}")
        rows = load(review)["rows"]
        accepted.extend((folder, review, row) for row in rows
                        if row["required_covered"] and all(row["required_covered"])
                        and row["claims"] and all(c["verdict"] == "supported" and not c["critical"]
                                                 for c in row["claims"]))
    numbered = REPORTS / "catalog" / "numbered-review.json"
    numbered_review = load(numbered)
    if numbered_review["responses_sha256"] != sha256(numbered.parent / "responses.jsonl"):
        raise ValueError(f"review changed: {numbered}")
    accepted.extend((numbered.parent, numbered, row) for row in numbered_review["cases"]
                    if row["answerable"] and row["verdict"] == "full")
    controls = REPORTS / "regressions-final" / "criteria.json"
    criteria = load(controls)
    if criteria["responses_sha256"] != sha256(controls.parent / "responses.jsonl"):
        raise ValueError(f"review changed: {controls}")
    accepted.extend((controls.parent, controls, row) for row in criteria["cases"]
                    if row["found"] and row["all_exact"] and all(row["checks"].values()))
    for study in ("completeness", "resolution"):
        for result in sorted((REPORTS / study / "series").glob("*/result.json")):
            scored = load(result)
            if (scored["responses_sha256"] != sha256(result.parent / "responses.jsonl")
                    or scored["manifest_sha256"] != sha256(result.parent / "manifest.json")):
                raise ValueError(f"review changed: {result}")
            accepted.extend((result.parent, result, row) for row in scored["requests"]
                            if row["expected_found"] and row["passed"])
    return accepted


def collect(gold: dict[str, dict]) -> dict[str, dict]:
    """Take the latest scored complete answer per exact question and document."""
    baselines: dict[str, dict] = {}
    sources = load(ROOT / "eval/gold/catalog-lock.json")["documents"]
    cached = {}
    for folder, review, row in previous_reviews():
        if folder not in cached:
            manifest = load(folder / "manifest.json")
            raw = folder / "responses.jsonl"
            if (manifest["result_type"] != "live"
                    or manifest["responses_sha256"] != sha256(raw)
                    or manifest["inputs"]["runtime"]["llm_digest"] != BALANCED_DIGEST
                    or manifest["inputs"]["runtime"]["settings"]["llm_thinking"] != "off"):
                raise ValueError(f"invalid prior measurement: {folder}")
            cached[folder] = manifest, {
                rec["req_id"]: rec for rec in load_jsonl(raw) if rec["kind"] == "gold"
            }
        manifest, responses = cached[folder]
        cid = row.get("req_id") or row["case_id"]
        case, response = gold[cid], responses[cid]
        fid = case["doc_fixture_id"]
        if (response["target_fixture"] != fid or response["question"] != case["question"]
                or manifest["inputs"]["documents"][fid]["source_sha256"] != sources[fid]
                or not case["answerable"] or response["http_status"] != 200
                or response["body"]["found"] is not True):
            raise ValueError(f"prior complete response does not match the locked question: {cid}")
        prior = baselines.get(cid)
        if prior is None or prior["finished_at"] < manifest["finished_at"]:
            baselines[cid] = {
                "fixture": fid,
                "question": case["question"],
                "previous_answer": response["body"]["answer"],
                "previous_commit": manifest["git_commit"], "finished_at": manifest["finished_at"],
                "responses": (folder / "responses.jsonl").relative_to(ROOT).as_posix(),
                "responses_sha256": manifest["responses_sha256"],
                "review": review.relative_to(ROOT).as_posix(), "review_sha256": sha256(review),
            }
    return dict(sorted(baselines.items()))


def packets(gold: dict[str, dict], prior: dict[str, dict], run: Path) -> tuple[list[dict], dict]:
    manifest = load(run / "manifest.json")
    responses_path = run / "responses.jsonl"
    if (manifest["responses_sha256"] != sha256(responses_path)
            or manifest["inputs"]["gold_sha256"] != sha256(WORK / "gold.jsonl")
            or manifest["inputs"]["runtime"]["llm_digest"] != BALANCED_DIGEST):
        raise ValueError("current response or gold does not match its manifest")
    metrics = {r["req_id"]: r for r in load_jsonl(run / "cases-all.jsonl")}
    responses = {r["req_id"]: r for r in load_jsonl(responses_path) if r["kind"] == "gold"}
    sources = {}
    rows = []
    for cid, baseline in prior.items():
        case, response, technical = gold[cid], responses[cid], metrics[cid]
        fid = baseline["fixture"]
        if response["target_fixture"] != fid or response["question"] != baseline["question"]:
            raise ValueError(f"current response differs from the question or document: {cid}")
        if fid not in sources:
            path = WORK / "index" / f"{fid}.json"
            if sha256(path) != manifest["inputs"]["documents"][fid]["export_sha256"]:
                raise ValueError(f"current index export changed: {fid}")
            sources[fid] = load(path)["canonical_text"]
        body = response["body"] if technical.get("schema_valid") else {}
        conditions = case.get("required_conditions", [])
        rows.append({
            "req_id": cid,
            "question": case["question"],
            "scope": case.get("scope") or str(case.get("section") or fid),
            "required_facts": case["expected_facts"],
            "condition_count": len(conditions),
            "known_errors": case["forbidden_facts"],
            "source_text": sources[fid],
            "evidence_spans": case.get("evidence", case["gold_spans"]),
            "citations": body.get("citations", []),
            "claims": segments(body["answer"]) if body.get("found") else [],
            "expected_found": True,
            "found": body.get("found"),
        })
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "review"))
    parser.add_argument("--run", default="final")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--judge", default="qwen3.6:27b-q4_K_M")
    args = parser.parse_args()
    gold = {r["case_id"]: r for r in load_jsonl(WORK / "gold.jsonl")}
    prior = collect(gold)
    run = WORK / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    plan = run / "preservation-plan.json"
    if plan.exists() and load(plan) != prior:
        raise ValueError("the archived baseline selection changed")
    write_json(plan, prior)
    print(f"{len(prior)} previously complete balanced answers", flush=True)
    if args.action == "plan":
        return
    rows, technical = packets(gold, prior, run)
    review_path = run / "preservation-review.json"
    review = (load(review_path) if review_path.exists() else
              {"reviewer": "", "reviewer_type": None, "rows": [blank_review(row) for row in rows]})
    model_review(rows, review, review_path, args.ollama, args.judge)
    labels = {label["req_id"]: label for label in review["rows"]}
    details = []
    for row in rows:
        cid = row["req_id"]
        selected = labels[cid]
        _, _, complete, critical, _ = check_review(row, selected)
        outcome = technical[cid]["outcome"]
        invalid_quotes = technical[cid].get("citation_problems", [])
        if complete and not invalid_quotes:
            outcome = "retained"
        elif outcome == "found":
            outcome = "degraded"
        details.append({
            "case_id": cid,
            "outcome": outcome,
            "critical": critical,
            "citation_problems": invalid_quotes,
            "missing_facts": [fact for fact, yes in
                              zip(row["required_facts"], selected["required_covered"], strict=True)
                              if not yes],
            "new_answer": " ".join(row["claims"]),
            "previous_answer": prior[cid]["previous_answer"],
            "previous_commit": prior[cid]["previous_commit"],
        })
    summary = {
        "reviewer": review["reviewer"],
        "baseline_sha256": sha256(plan),
        "responses_sha256": sha256(run / "responses.jsonl"),
        "counts": {outcome: sum(row["outcome"] == outcome for row in details)
                   for outcome in sorted({row["outcome"] for row in details})},
        "cases": details,
    }
    write_json(run / "preservation.json", summary)
    print(summary["counts"], flush=True)


if __name__ == "__main__":
    main()
