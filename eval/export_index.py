"""Host wrapper: export canonical text + published chunks of every indexed fixture.

    python3 eval/export_index.py [--project b1-eval]

Runs eval/incontainer/export_index.py inside <project>-api-1 (read-only) and
writes eval/work/index/<fixture>.json plus eval/work/canonical/<fixture>.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="b1-eval")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    # Mapping from the public API (title = file stem -> fixture id prefix), so the
    # export also works while the rest of the corpus is still being indexed.
    with urllib.request.urlopen(f"{args.api}/api/v1/documents?limit=100") as resp:
        items = json.load(resp)["items"]
    by_doc = {i["doc_id"]: (i["title"] or "").split("_", 1)[0] for i in items
              if i["status"] == "succeeded"}
    by_doc = {d: f for d, f in by_doc.items() if f != "100"}  # timing-only fixture
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
