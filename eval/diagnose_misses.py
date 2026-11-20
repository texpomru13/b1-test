"""Diagnose answerable cases that ended found=false: at which QA step did the repeated request stop?

    uv run python eval/diagnose_misses.py --run balanced-9b-nothink [--project b1-eval]

Re-asks each missed answerable case once (debug=true) and collects the API log
events emitted during that request (draft_invalid / provenance_rejected /
verifier_invalid / question_answered). Diagnosis only: nothing is tuned on these
cases. Writes eval/work/runs/<run>/miss_diagnosis.jsonl and prints a breakdown.
Note: a re-ask is a new sample (the model is not deterministic), so the
outcome may differ from the original run; both are recorded.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import time

import httpx
from paths import WORK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--project", default="b1-eval")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    run_dir = WORK / "runs" / args.run
    cases = []
    for split in ("dev", "holdout"):
        rows = [json.loads(x) for x in (run_dir / f"cases-{split}.jsonl").read_text().splitlines()]
        cases += [r for r in rows if r["kind"] == "gold" and r["answerable"]
                  and r["outcome"] == "not_found"]
    responses = {json.loads(x)["req_id"]: json.loads(x)
                 for x in (run_dir / "responses.jsonl").read_text().splitlines() if x}
    out_path = run_dir / "miss_diagnosis.jsonl"
    client = httpx.Client(base_url=args.api, timeout=900)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in cases:
            rec = responses[row["req_id"]]
            # unix seconds: an RFC3339 string without a zone is read as LOCAL time
            since = str(int(time.time()) - 1)
            resp = client.post(f"/api/v1/documents/{rec['doc_id']}/questions",
                               json={"question": rec["question"], "debug": True})
            request_id = resp.headers["X-Request-ID"]
            time.sleep(1.0)
            # structlog writes to the container's stderr: merge both streams
            logs = subprocess.run(["docker", "logs", "--since", since, f"{args.project}-api-1"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True).stdout.splitlines()
            events = []
            for line in logs:
                if line.startswith("{"):
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("request_id") != request_id:
                        continue
                    if e.get("event") in ("draft_invalid", "provenance_rejected",
                                          "verifier_invalid", "question_answered",
                                          "llm_outcome_uncertain", "verifier_verdict",
                                          "draft_not_answerable"):
                        events.append({k: e.get(k) for k in ("event", "error", "found",
                                                             "llm_calls", "repairs", "coverage",
                                                             "verdicts") if k in e})
            body = resp.json()
            usage = (body.get("diagnostics") or {}).get("token_usage", {})
            final = next((e for e in events if e["event"] == "question_answered"), {})
            decisions = [e for e in events if e["event"] in (
                "draft_not_answerable", "provenance_rejected", "draft_invalid", "verifier_invalid"
            )]
            if resp.status_code != 200:
                stage = f"http_{resp.status_code}"
            elif body.get("found"):
                stage = "answered_on_reask"
            elif decisions:
                last = decisions[-1]
                stage = last.get("error") or last["event"]
            else:
                stage = "unclassified"
            fh.write(json.dumps({"req_id": row["req_id"], "request_id": request_id,
                                 "category": row["category"],
                                 "ctx_all": row.get("ctx_all"), "reask_http": resp.status_code,
                                 "reask_found": body.get("found"),
                                 "llm_calls": usage.get("llm_calls"),
                                 "repairs": final.get("repairs"), "events": events, "stage": stage},
                                ensure_ascii=False) + "\n")
            print(row["req_id"], stage, flush=True)
    rows = [json.loads(x) for x in out_path.read_text().splitlines()]
    print(json.dumps(collections.Counter(r["stage"] for r in rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
