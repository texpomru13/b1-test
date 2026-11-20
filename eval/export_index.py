"""Host wrapper: export canonical text + published chunks of every indexed fixture.

    python3 eval/export_index.py [--project b1-eval]

Runs eval/incontainer/export_index.py inside <project>-api-1 (read-only) and
writes eval/work/index/<fixture>.json plus eval/work/canonical/<fixture>.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from paths import ROOT, WORK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="b1-eval")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    corpus = json.loads((WORK / "corpus.json").read_text(encoding="utf-8"))
    if corpus["api"] != args.api:
        parser.error("corpus.json belongs to a different API, upload the corpus first")
    by_doc = {item["doc_id"]: fid for fid, item in corpus["docs"].items()
              if item.get("status") == "succeeded" and fid != "100_pages"}
    script = (ROOT / "eval" / "incontainer" / "export_index.py").read_text()
    result = subprocess.run(
        ["docker", "exec", "-i", "-e", f"EVAL_DOC_IDS={','.join(by_doc)}",
         f"{args.project}-api-1", "python", "-"],
        input=script, capture_output=True, text=True, check=True,
    )
    exported = json.loads(result.stdout)
    (WORK / "index").mkdir(parents=True, exist_ok=True)
    (WORK / "canonical").mkdir(parents=True, exist_ok=True)
    for doc_id, payload in exported.items():
        fid = by_doc[doc_id]
        if payload.get("error"):
            raise SystemExit(f"Could not export {fid}: {payload['error']}")
        payload["doc_id"] = doc_id
        payload["fixture_id"] = fid
        (WORK / "index" / f"{fid}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if "canonical_text" in payload:
            (WORK / "canonical" / f"{fid}.txt").write_text(
                payload["canonical_text"], encoding="utf-8")
        print(f"{fid}: chunks={len(payload.get('chunks', []))} "
              f"chars={len(payload.get('canonical_text', ''))} {payload.get('error', '')}")


if __name__ == "__main__":
    main()
