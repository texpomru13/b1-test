"""Capture model quotes that fail the exact-substring provenance check (INSIDE api container).

Env EVAL_QUERIES_JSON = [{"req_id", "doc_id", "question"}]. Runs the production QA
graph unchanged, but wraps grounding.locate_quote (graph-module reference) to record
each rejected quote together with the closest chunk window, and classifies the
mismatch: whitespace-insensitive match (PDF intra-word spaces), quote-mark / dash /
ё normalization, or genuinely paraphrased. Output: JSON on stdout. Diagnosis only.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import sys

import app.qa.grounding as grounding
from app.config.settings import get_settings
from app.inference.client import LocalModelsClient
from app.ingestion.chunker import ApproxTokenLength, HFTokenizerLength
from app.providers.factory import build_chat_adapter
from app.qa.graph import QaDependencies, build_qa_graph, run_qa
from app.reliability.deadline import Deadline
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import RetrievalScope
from app.storage.qdrant_repo import create_qdrant_client
from app.storage.redis_repo import RedisJobRepository, create_redis_client

FAILS: list[dict] = []
_orig = grounding.locate_quote
_TRANS = str.maketrans({"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "—": "-",
                        "–": "-", "ё": "е", "Ё": "Е", " ": " "})


def _classify(chunk_text: str, quote: str) -> str:
    squash = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    if squash(quote) in squash(chunk_text):
        return "whitespace_only (intra-word spaces in PDF text)"
    if squash(quote.translate(_TRANS)) in squash(chunk_text.translate(_TRANS)):
        return "punctuation/yo normalization"
    return "paraphrased_or_altered"


def wrapped(chunk, quote):  # noqa: ANN001, ANN202
    try:
        return _orig(chunk, quote)
    except grounding.ProvenanceError:
        text = chunk.text
        sm = difflib.SequenceMatcher(None, text, quote, autojunk=False)
        m = sm.find_longest_match(0, len(text), 0, len(quote))
        lo = max(0, m.a - m.b - 20)
        FAILS.append({"quote": quote[:300], "chunk_window": text[lo:lo + len(quote) + 40][:340],
                      "kind": _classify(text, quote)})
        raise


grounding.locate_quote = wrapped  # validate_claims resolves it from the module


async def main() -> None:
    s = get_settings()
    repo = RedisJobRepository(create_redis_client(s.redis_url))
    length_fn = (HFTokenizerLength(__import__("pathlib").Path(s.embedding_tokenizer_path))
                 if s.embedding_tokenizer_path else ApproxTokenLength())
    deps = QaDependencies(
        settings=s, qdrant=create_qdrant_client(s.qdrant_url, timeout_s=60),
        local_models=LocalModelsClient(s.local_models_url, timeout_s=600),
        bm25=Bm25Adapter(language=s.bm25_language, avg_len=float(s.bm25_avg_len),
                         cache_dir=s.fastembed_cache_dir),
        length_fn=length_fn, adapter=build_chat_adapter(s),
    )
    graph = build_qa_graph(deps)
    out = []
    for q in json.loads(os.environ["EVAL_QUERIES_JSON"]):
        job = await repo.get_job(await repo.get_job_id_for_doc(q["doc_id"]))
        scope = RetrievalScope(q["doc_id"], job["accepted_run_id"], job["index_fingerprint"])
        FAILS.clear()
        state = await run_qa(graph, question=q["question"], scope=scope, debug=False,
                             deadline=Deadline(total_s=float(s.qa_deadline_s)))
        out.append({"req_id": q["req_id"], "outcome": state.get("outcome"),
                    "rejected_quotes": list(FAILS)})
        print(q["req_id"], state.get("outcome"), [f["kind"] for f in FAILS],
              file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, ensure_ascii=False)


asyncio.run(main())
