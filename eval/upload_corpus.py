"""Upload the evaluation corpus through the public API and time background indexing.

Run with the app environment (httpx + pypdf):  uv run python eval/upload_corpus.py

- Uploads every file in test_documents/ plus a 100-page fixture assembled from
  real pages of the corpus (eval/work/fixtures/100_pages.pdf, never committed).
- Polls GET /api/v1/jobs/{job_id} until a terminal state; records upload latency
  (HTTP must NOT wait for indexing), time-to-terminal, page/chunk counts, stages.
- Writes eval/work/corpus.json (fixture_id -> doc_id mapping used by every later
  step). Idempotent: an existing mapping for the same API base is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "test_documents"
WORK = ROOT / "eval" / "work"
TERMINAL = {"succeeded", "failed", "cleanup_pending"}
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def fixture_id(path: Path) -> str:
    return path.stem.split("_", 1)[0] if path.parent == DOCS else path.stem


def build_100_pages() -> Path:
    """100 real, mutually different pages (07 + 14 + 02 + 15) -> one PDF."""
    out = WORK / "fixtures" / "100_pages.pdf"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for name in ("07_", "14_", "02_", "15_"):
        src = next(DOCS.glob(f"{name}*.pdf"))
        for page in PdfReader(src).pages:
            if len(writer.pages) >= 100:
                break
            writer.add_page(page)
    with out.open("wb") as fh:
        writer.write(fh)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--only", nargs="*", help="fixture ids to (re)upload")
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    corpus_path = WORK / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else {}
    if corpus.get("api") != args.api:
        corpus = {"api": args.api, "docs": {}}

    files = sorted(p for p in DOCS.iterdir() if p.suffix in CONTENT_TYPES)
    files.append(build_100_pages())
    client = httpx.Client(base_url=args.api, timeout=120)

    pending: dict[str, dict] = {}
    for path in files:
        fid = fixture_id(path)
        if args.only and fid not in args.only:
            continue
        if fid in corpus["docs"] and corpus["docs"][fid].get("status") == "succeeded":
            continue
        data = path.read_bytes()
        started = time.monotonic()
        resp = client.post(
            "/api/v1/documents",
            files={"file": (path.name, data, CONTENT_TYPES[path.suffix])},
            data={"title": path.stem},
        )
        upload_ms = int((time.monotonic() - started) * 1000)
        resp.raise_for_status()
        body = resp.json()
        entry = {
            "file": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "doc_id": body["doc_id"],
            "job_id": body["job_id"],
            "upload_http_ms": upload_ms,
            "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "_t0": time.monotonic() - upload_ms / 1000,
            "stages_seen": [],
        }
        corpus["docs"][fid] = entry
        pending[fid] = entry
        print(f"uploaded {fid}: http {resp.status_code} in {upload_ms} ms")

    # Poll all jobs (the worker processes them from the queue in order).
    deadline = time.monotonic() + args.timeout_s
    while pending and time.monotonic() < deadline:
        for fid, entry in list(pending.items()):
            job = client.get(f"/api/v1/jobs/{entry['job_id']}").json()
            stage = f"{job['status']}:{job['stage']}"
            if not entry["stages_seen"] or entry["stages_seen"][-1][0] != stage:
                entry["stages_seen"].append(
                    (stage, round(time.monotonic() - entry["_t0"], 1))
                )
            if job["status"] in TERMINAL:
                entry["status"] = job["status"]
                entry["error"] = job.get("error")
                entry["terminal_s"] = round(time.monotonic() - entry["_t0"], 1)
                detail = client.get(f"/api/v1/documents/{entry['doc_id']}").json()
                entry["page_count"] = detail.get("page_count")
                entry["chunk_count"] = detail.get("chunk_count")
                print(f"{fid}: {job['status']} after {entry['terminal_s']} s "
                      f"pages={entry['page_count']} chunks={entry['chunk_count']} "
                      f"error={(job.get('error') or {}).get('code')}")
                pending.pop(fid)
        time.sleep(1.0)
    for entry in corpus["docs"].values():
        entry.pop("_t0", None)
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=1))
    if pending:
        raise SystemExit(f"timed out waiting for: {sorted(pending)}")


if __name__ == "__main__":
    main()
