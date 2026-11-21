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
from catalog import prepare_gold
from paths import ROOT, WORK
from pypdf import PdfReader, PdfWriter
from qualification_data import install_gold, source_files

DOCS = ROOT / "test_documents"
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--catalog", action="store_true",
                       help="use all real documents and questions")
    group.add_argument("--qualification", action="store_true",
                        help="use the locked independent corpus in a separate B1_EVAL_WORK_DIR")
    args = parser.parse_args()
    if args.qualification:
        install_gold()

    WORK.mkdir(parents=True, exist_ok=True)
    corpus_path = WORK / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else {}
    if corpus.get("api") != args.api:
        corpus = {"api": args.api, "docs": {}}

    if args.catalog:
        fixtures = prepare_gold()
    elif args.qualification:
        fixtures = source_files()
    else:
        files = sorted(p for p in DOCS.iterdir() if p.suffix in CONTENT_TYPES)
        files.append(build_100_pages())
        fixtures = {fixture_id(p): p for p in files}
    client = httpx.Client(base_url=args.api, timeout=120)

    def checkpoint() -> None:
        state = {fid: {k: v for k, v in entry.items() if k != "_t0"}
                 for fid, entry in corpus["docs"].items()}
        tmp = corpus_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**corpus, "docs": state}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(corpus_path)

    pending: dict[str, dict] = {}
    for fid, path in fixtures.items():
        if args.only and fid not in args.only:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        previous = corpus["docs"].get(fid, {})
        if previous.get("status") == "succeeded" and previous.get("sha256") == digest:
            detail = client.get(f"/api/v1/documents/{previous['doc_id']}")
            if detail.status_code != 404:
                detail.raise_for_status()
                if detail.json().get("ready"):
                    continue
        if (previous.get("sha256") == digest and previous.get("job_id")
                and previous.get("status") not in TERMINAL):
            saved = client.get(f"/api/v1/jobs/{previous['job_id']}")
            if saved.status_code != 404:
                saved.raise_for_status()
                elapsed = datetime.now(UTC) - datetime.fromisoformat(previous["submitted_at"])
                previous["_t0"] = time.monotonic() - elapsed.total_seconds()
                pending[fid] = previous
                continue
        retry_headers = {}
        if previous.get("sha256") == digest and previous.get("status") == "failed":
            retry_headers["Idempotency-Key"] = f"eval-retry-{previous['job_id']}"
        started = time.monotonic()
        resp = client.post(
            "/api/v1/documents",
            files={"file": (path.name, data, CONTENT_TYPES[path.suffix])},
            data={"title": path.stem}, headers=retry_headers,
        )
        upload_ms = int((time.monotonic() - started) * 1000)
        resp.raise_for_status()
        body = resp.json()
        entry = {
            "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            "sha256": digest,
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
        checkpoint()
        print(f"uploaded {fid}: http {resp.status_code} in {upload_ms} ms")

    # Poll all jobs (the worker processes them from the queue in order).
    deadline = time.monotonic() + args.timeout_s
    while pending and time.monotonic() < deadline:
        for fid, entry in list(pending.items()):
            job_response = client.get(f"/api/v1/jobs/{entry['job_id']}")
            job_response.raise_for_status()
            job = job_response.json()
            stage = f"{job['status']}:{job['stage']}"
            if not entry["stages_seen"] or entry["stages_seen"][-1][0] != stage:
                entry["stages_seen"].append(
                    (stage, round(time.monotonic() - entry["_t0"], 1))
                )
            if job["status"] in TERMINAL:
                entry["status"] = job["status"]
                entry["error"] = job.get("error")
                entry["terminal_s"] = round(time.monotonic() - entry["_t0"], 1)
                detail_response = client.get(f"/api/v1/documents/{entry['doc_id']}")
                detail_response.raise_for_status()
                detail = detail_response.json()
                entry["page_count"] = detail.get("page_count")
                entry["chunk_count"] = detail.get("chunk_count")
                print(f"{fid}: {job['status']} after {entry['terminal_s']} s "
                      f"pages={entry['page_count']} chunks={entry['chunk_count']} "
                      f"error={(job.get('error') or {}).get('code')}")
                pending.pop(fid)
                checkpoint()
        if pending:
            time.sleep(1.0)
    checkpoint()
    if pending:
        raise SystemExit(f"timed out waiting for: {sorted(pending)}")
    for fid, entry in corpus["docs"].items():
        if args.only and fid not in args.only:
            continue
        if entry["status"] == "succeeded" and (fid != "100_pages" or entry["page_count"] == 100):
            continue
        raise SystemExit(f"unexpected result for {fid}: {entry['status']} {entry['error']}")


if __name__ == "__main__":
    main()
