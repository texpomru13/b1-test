"""Export canonical text + published chunks for evaluation (runs INSIDE the api container).

    docker exec -i <project>-api-1 python - < eval/incontainer/export_index.py \
        < (doc ids JSON on the first stdin line is not possible with '-') ...

Usage (see eval/export_index.sh): the doc ids are passed via the EVAL_DOC_IDS env
var (comma-separated). Output: one JSON object on stdout:
  {doc_id: {accepted_run_id, index_fingerprint, canonical_text, pages, chunks[]}}
Read-only: Redis job registry, the parse artifact on the documents volume and a
scoped Qdrant scroll of the ACCEPTED run (same scope filter the QA path uses).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.retrieval.search import RetrievalScope, scope_filter
from app.storage.qdrant_repo import create_qdrant_client
from app.storage.redis_repo import RedisJobRepository, create_redis_client


async def main() -> None:
    settings = get_settings()
    repo = RedisJobRepository(create_redis_client(settings.redis_url))
    qdrant = create_qdrant_client(settings.qdrant_url, timeout_s=60)
    out: dict[str, dict] = {}
    for doc_id in [d for d in os.environ["EVAL_DOC_IDS"].split(",") if d]:
        job_id = await repo.get_job_id_for_doc(doc_id)
        job = await repo.get_job(job_id) if job_id else None
        if not job or not job.get("accepted_run_id"):
            out[doc_id] = {"error": f"no accepted run (status={job and job.get('status')})"}
            continue
        scope = RetrievalScope(doc_id, job["accepted_run_id"], job["index_fingerprint"])
        parse_dir = Path(settings.documents_dir) / doc_id / "parse"
        parsed = json.loads((parse_dir / "parsed.json").read_text(encoding="utf-8"))
        canonical = (parse_dir / "canonical.txt").read_text(encoding="utf-8")
        chunks = []
        offset = None
        while True:
            points, offset = await qdrant.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=scope_filter(scope), limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                chunks.append({
                    "chunk_id": str(p.id),
                    "chunk_index": pl.get("chunk_index"),
                    "char_start": pl.get("char_start"),
                    "char_end": pl.get("char_end"),
                    "page_from": pl.get("page_from"),
                    "page_to": pl.get("page_to"),
                    "section": pl.get("section"),
                    "text": pl.get("text"),
                })
            if offset is None:
                break
        chunks.sort(key=lambda c: c["chunk_index"])
        out[doc_id] = {
            "accepted_run_id": job["accepted_run_id"],
            "index_fingerprint": job["index_fingerprint"],
            "canonical_text": canonical,
            "canonical_sha256": parsed.get("canonical_sha256"),
            "pages": parsed.get("pages"),
            "chunk_tokens": parsed.get("chunk_tokens"),
            "overlap_tokens": parsed.get("overlap_tokens"),
            "length_fn": parsed.get("length_fn"),
            "chunks": chunks,
        }
    await qdrant.close()
    await repo.aclose()
    json.dump(out, sys.stdout, ensure_ascii=False)


asyncio.run(main())
