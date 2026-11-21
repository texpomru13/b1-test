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
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from paths import WORK, gold_path
from provenance import environment, live_inputs, sha256, write_json


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
    parser.add_argument("--project", help="verify images and models of this Compose project")
    parser.add_argument("--case", nargs="+", help="run only these request IDs (includes iso- IDs)")
    parser.add_argument("--no-isolation", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=960.0)
    args = parser.parse_args()

    gold = load_jsonl(gold_path())
    fixtures = sorted({c["doc_fixture_id"] for c in gold})
    doc_ids = {fid: json.loads((WORK / "index" / f"{fid}.json").read_text())["doc_id"]
               for fid in fixtures}
    reqs = build_requests(gold, fixtures, args.split)
    if args.no_isolation:
        reqs = [r for r in reqs if r["kind"] == "gold"]

    if args.case:
        unknown = set(args.case) - {r["req_id"] for r in reqs}
        if unknown:
            parser.error(f"unknown request IDs: {sorted(unknown)}")
        reqs = [r for r in reqs if r["req_id"] in args.case]
    run_dir = WORK / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "responses.jsonl"
    previous = load_jsonl(out_path)
    done = {r["req_id"]: r for r in previous}
    if len(previous) != len(done):
        parser.error("duplicate request IDs in saved responses")
    for req in reqs:
        saved = done.get(req["req_id"])
        if saved and any(saved.get(key) != value for key, value in req.items()):
            parser.error(f"saved question or target differs for {req['req_id']}; use a new --run")
    client = httpx.Client(base_url=args.api, timeout=args.timeout_s)

    meta_path = run_dir / "meta.json"
    manifest_path = run_dir / "manifest.json"
    inputs = live_inputs(client, args.project, doc_ids)
    ready = client.get("/health/ready")
    ready.raise_for_status()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["api"] != args.api or meta.get("inputs") != inputs:
            parser.error("the API, build, settings, documents or gold changed; use a new --run")
    else:
        if previous or manifest_path.exists():
            parser.error("saved responses are missing their provenance; use a new --run")
        meta = {"run": args.run, "api": args.api, "started_at": datetime.now(UTC).isoformat(),
                "initial_split": args.split, "doc_ids": doc_ids, "ready": ready.json(),
                "inputs": inputs, **environment()}
        write_json(meta_path, meta)
    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text())
        if saved_manifest.get("responses_sha256") != sha256(out_path):
            parser.error("saved responses differ from their manifest; use a new --run")

    todo = [r for r in reqs if r["req_id"] not in done]
    print(f"{len(reqs)} requests, {len(done)} already done, {len(todo)} to run")
    if todo:
        manifest_path.unlink(missing_ok=True)  # interrupted runs keep their metadata and raw rows
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
                record["request_id"] = resp.headers.get("X-Request-ID")
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

    if live_inputs(client, args.project, doc_ids) != inputs:
        parser.error("the running build or inputs changed during this run; discard this result")
    write_json(manifest_path, {
        **meta, "result_type": "live", "finished_at": datetime.now(UTC).isoformat(),
        "response_count": len(done) + len(todo), "responses_sha256": sha256(out_path),
    })
    client.close()


if __name__ == "__main__":
    main()
