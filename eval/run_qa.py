"""Run the gold questions (and cross-document isolation probes) through the PUBLIC API.

    uv run python eval/run_qa.py --run balanced-9b [--split all|dev|holdout]

Every request uses debug=true, so diagnostics (fused candidates with rerank
scores, context ids, cited ids, stage timings, model ids) are recorded together
with the raw HTTP status/body and the client-side wall time. Sequential on
purpose: the API admits one question at a time (MAX_CONCURRENT_QA) and latency
percentiles must not include queueing. Resumable: finished case ids are skipped.

Isolation probes: each entity-specific answerable case is ALSO asked against a
different document (deterministic rotation); the expected outcome is found=false
and every citation (if any) must come from the queried document.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_requests(gold: list[dict], fixtures: list[str], split: str) -> list[dict]:
    reqs = []
    for case in gold:
        if split != "all" and case["split"] != split:
            continue
        reqs.append({"req_id": case["case_id"], "case_id": case["case_id"],
                     "kind": "gold", "target_fixture": case["doc_fixture_id"],
                     "question": case["question"]})
    ordered = sorted(fixtures)
    for case in gold:
        if split != "all" and case["split"] != split:
            continue
        if not (case["answerable"] and case.get("entity_specific")):
            continue
        idx = ordered.index(case["doc_fixture_id"])
        other = ordered[(idx + 1) % len(ordered)]
        reqs.append({"req_id": f"iso-{case['case_id']}->{other}", "case_id": case["case_id"],
                     "kind": "isolation", "target_fixture": other,
                     "question": case["question"]})
    return reqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name, e.g. balanced-9b")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=960.0)
    args = parser.parse_args()

    gold = load_jsonl(ROOT / "eval" / "gold" / "gold.jsonl")
    fixtures = sorted({c["doc_fixture_id"] for c in gold})
    doc_ids = {fid: json.loads((WORK / "index" / f"{fid}.json").read_text())["doc_id"]
               for fid in fixtures}
    reqs = build_requests(gold, fixtures, args.split)
    if args.no_isolation:
        reqs = [r for r in reqs if r["kind"] == "gold"]

    run_dir = WORK / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "responses.jsonl"
    done = {r["req_id"] for r in load_jsonl(out_path)}
    client = httpx.Client(base_url=args.api, timeout=args.timeout_s)

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
        meta_path.write_text(json.dumps({
            "run": args.run, "api": args.api, "started_at": datetime.now(UTC).isoformat(),
            "git_head": sha, "split": args.split,
            "ready": client.get("/health/ready").json(),
        }, ensure_ascii=False, indent=1))

    todo = [r for r in reqs if r["req_id"] not in done]
    print(f"{len(reqs)} requests, {len(done)} already done, {len(todo)} to run")
    with out_path.open("a", encoding="utf-8") as fh:
        for n, req in enumerate(todo, 1):
            doc_id = doc_ids[req["target_fixture"]]
            started = time.monotonic()
            record = dict(req, doc_id=doc_id,
                          asked_at=datetime.now(UTC).isoformat(timespec="seconds"))
            try:
                resp = client.post(f"/api/v1/documents/{doc_id}/questions",
                                   json={"question": req["question"], "debug": True})
                record["http_status"] = resp.status_code
                try:
                    record["body"] = resp.json()
                except ValueError:
                    record["body"] = {"_raw": resp.text[:2000]}
            except httpx.HTTPError as exc:
                record["http_status"] = None
                record["body"] = {"_client_error": f"{type(exc).__name__}: {exc}"}
            record["wall_ms"] = int((time.monotonic() - started) * 1000)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            body = record["body"] if isinstance(record["body"], dict) else {}
            print(f"[{n}/{len(todo)}] {req['req_id']}: {record['http_status']} "
                  f"found={body.get('found')} {record['wall_ms']} ms", flush=True)


if __name__ == "__main__":
    main()
