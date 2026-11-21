"""Repeat live generation and verification over the thirteen archived retrieval results.

    .venv/bin/python eval/replay_refusals.py --output /tmp/new-refusal-replay.jsonl

Runs sequentially against the local Ollama model named in the manifest. The
published contexts include the original chunk order, canonical offsets, pages
and recovered list-heading links. Retrieval and the HTTP route are bypassed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from paths import ROOT

sys.path.insert(0, str(ROOT))
from app.config.settings import Settings
from app.ingestion.chunker import ApproxTokenLength
from app.providers.factory import build_chat_adapter
from app.qa.assemble import assemble_answer
from app.qa.workflow import OUTCOME_ANSWERED, QaDependencies, run_qa
from app.retrieval.search import ContextChunk, Diagnostics, RetrievalResult, RetrievalScope

REPORT = ROOT / "eval/reports/refusal-replay"


async def replay(output: Path, ollama_url: str) -> None:
    manifest = json.loads((REPORT / "manifest.json").read_text())
    inputs = REPORT / "contexts.jsonl"
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != manifest["contexts_sha256"]:
        raise ValueError("archived contexts differ from manifest")
    async with httpx.AsyncClient() as client:
        response = await client.get(ollama_url + "/api/tags")
        response.raise_for_status()
        tags = {m["name"]: m["digest"] for m in response.json()["models"]}
        if tags.get(manifest["model"]["name"]) != manifest["model"]["digest"]:
            raise ValueError("the archived Ollama model digest is not present")
    settings = Settings(_env_file=None, ollama_base_url=ollama_url, llm_provider="ollama",
                        llm_model=manifest["model"]["name"], **manifest["settings"])
    adapter = build_chat_adapter(settings)
    # hybrid_search is the only substituted call. The sampled draft, source ID
    # resolution, deterministic checks and LLM verification all use current code.
    deps = QaDependencies(settings, None, None, None, ApproxTokenLength(), adapter)
    with output.open("x", encoding="utf-8") as records:
        for line in inputs.read_text().splitlines():
            case = json.loads(line)
            context = [ContextChunk(**chunk) for chunk in case["context"]]
            retrieval = RetrievalResult(context, [c.chunk_id for c in context], Diagnostics())
            scope = RetrievalScope(case["doc_id"], case["ingestion_run_id"],
                                   case["index_fingerprint"])
            with patch("app.qa.workflow.hybrid_search", AsyncMock(return_value=retrieval)):
                state = await run_qa(deps, question=case["question"], scope=scope, debug=True)
            body = (assemble_answer(state).model_dump(exclude_none=True)
                    if state.outcome == OUTCOME_ANSWERED else {"found": False, "citations": []})
            record = {"case_id": case["case_id"], "question": case["question"],
                      "outcome": state.outcome, "llm_calls": state.llm_calls,
                      "refusal_stage": state.refusal_stage, "body": body}
            records.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.flush()
            print(case["case_id"], state.outcome, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    asyncio.run(replay(args.output, args.ollama_url))
