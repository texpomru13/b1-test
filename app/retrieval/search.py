"""Hybrid retrieval with a mandatory reranker (plan §6, ADR 004).

- ONE Qdrant query_points call builds the fused candidate list: two prefetch
  branches (dense + bm25), the immutable scope filter in BOTH branches AND at the
  root, server-side fusion (RRF default / DBSF configurable). No Python merging.
- The reranker is mandatory and fails CLOSED: its unavailability is an error,
  never a dense-only answer.
- Context budget is computed for the target LLM with a separate reasoning
  reserve (§6.3): whole chunks are selected in rerank order until either
  CONTEXT_TOP_K or the token budget is reached; everything not selected is
  excluded from the allowed spans (no partial chunk truncation).
- debug=true adds a diagnostic query_batch_points REPLAY on the same published
  run (report-only; never feeds candidates) — ADR 004.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.settings import FusionMethod, Settings
from app.inference.client import LocalModelsClient, LocalModelsError
from app.inference.contract import (
    EmbedRequest,
    InputType,
    RerankDocument,
    RerankRequest,
)
from app.ingestion.chunker import LengthFn
from app.observability.logging import get_logger
from app.reliability.policy import retry_async
from app.retrieval.bm25 import Bm25Adapter
from app.storage.qdrant_repo import DENSE_NAME, SPARSE_NAME
from qdrant_client import AsyncQdrantClient, models

logger = get_logger("retrieval.search")

# Conservative token estimates for parts that do not exist until stage 6 wires
# the real prompts; re-measured there (§6.3: the real limit is recomputed per call).
SYSTEM_AND_SCHEMA_RESERVE_TOKENS = 1500
SAFETY_MARGIN_TOKENS = 512

ERROR_RERANKER_UNAVAILABLE = "RERANKER_UNAVAILABLE"
ERROR_EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
ERROR_QDRANT_UNAVAILABLE = "QDRANT_UNAVAILABLE"


class RetrievalError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class RetrievalScope:
    doc_id: str
    accepted_run_id: str
    index_fingerprint: str


def scope_filter(scope: RetrievalScope) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=scope.doc_id)),
            models.FieldCondition(
                key="ingestion_run_id",
                match=models.MatchValue(value=scope.accepted_run_id),
            ),
            models.FieldCondition(
                key="index_fingerprint",
                match=models.MatchValue(value=scope.index_fingerprint),
            ),
        ]
    )


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str  # qdrant point id (server-issued, used as the citation source id)
    chunk_index: int
    text: str
    section: str | None
    page_from: int
    page_to: int
    char_start: int
    char_end: int
    rerank_score: float
    # Per-page spans for chunks crossing page boundaries: [{page, char_start, char_end}]
    segments: list[dict[str, int]] = field(default_factory=list)


@dataclass
class Diagnostics:
    dense_returned: int | None = None
    sparse_returned: int | None = None
    fused_returned: int = 0
    reranked_count: int = 0
    context_count: int = 0
    context_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] | None = None  # filled by QA (stage 6); never mixed
    rerank_scores: dict[str, float] = field(default_factory=dict)
    stage_ms: dict[str, int] = field(default_factory=dict)
    model_ids: dict[str, str] = field(default_factory=dict)
    branch_counts_source: str | None = None
    diagnostic_warnings: list[str] = field(default_factory=list)
    dropped_by_budget_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    context: list[ContextChunk]
    allowed_chunk_ids: list[str]  # ONLY these spans may be cited (post-truncation)
    diagnostics: Diagnostics


def effective_context_budget(settings: Settings, question_tokens: int) -> int:
    """§6.3: model_context - system/question/schema - reasoning - output - margin."""
    computed = (
        settings.local_context_tokens
        - SYSTEM_AND_SCHEMA_RESERVE_TOKENS
        - question_tokens
        - settings.llm_reasoning_budget_tokens
        - settings.llm_max_output_tokens
        - SAFETY_MARGIN_TOKENS
    )
    return max(0, min(settings.context_budget_tokens, computed))


async def hybrid_search(
    *,
    question: str,
    scope: RetrievalScope,
    settings: Settings,
    qdrant: AsyncQdrantClient,
    local_models: LocalModelsClient,
    bm25: Bm25Adapter,
    length_fn: LengthFn,
    debug: bool = False,
    freeze_dir: Path | None = None,
) -> RetrievalResult:
    import asyncio

    diag = Diagnostics()
    flt = scope_filter(scope)
    timer = time.monotonic()

    # 1) encode the question (dense via local-models; BM25 off the event loop)
    try:
        dense_response = await local_models.embed(
            EmbedRequest(texts=[question], input_type=InputType.QUERY)
        )
    except LocalModelsError as exc:
        raise RetrievalError(
            ERROR_EMBEDDING_UNAVAILABLE, f"query embedding failed: {exc}"
        ) from exc
    dense_query = dense_response.vectors[0]
    sparse_query = await asyncio.to_thread(bm25.embed_query, question)
    diag.model_ids["embedding"] = dense_response.model_id
    diag.stage_ms["encode"] = int((time.monotonic() - timer) * 1000)

    # 2) THE single fused candidates query (server-side fusion)
    timer = time.monotonic()
    fusion = (
        models.Fusion.RRF
        if settings.fusion_method == FusionMethod.RRF
        else models.Fusion.DBSF
    )

    async def fused_query() -> Any:
        # Still ONE Query API call per attempt: both prefetch branches and the
        # fusion run server-side. A read-only query is safe to repeat.
        return await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_query, using=DENSE_NAME,
                    filter=flt, limit=settings.dense_top_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices, values=sparse_query.values
                    ),
                    using=SPARSE_NAME, filter=flt, limit=settings.sparse_top_k,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            query_filter=flt,
            limit=settings.rerank_input_k,
            with_payload=True,
            with_vectors=False,
        )

    try:
        # Bounded retry on a temporarily unavailable vector store (§8.6).
        fused = await retry_async(
            fused_query, op_name="qdrant.query_points",
            max_attempts=settings.qdrant_max_attempts,
        )
    except Exception as exc:
        raise RetrievalError(
            ERROR_QDRANT_UNAVAILABLE, f"hybrid query failed: {type(exc).__name__}"
        ) from exc
    candidates = fused.points
    diag.fused_returned = len(candidates)
    diag.stage_ms["fusion"] = int((time.monotonic() - timer) * 1000)

    if freeze_dir is not None:
        _freeze_candidates(freeze_dir, question, scope, candidates)

    if not candidates:
        diag.reranked_count = 0
        return RetrievalResult(context=[], allowed_chunk_ids=[], diagnostics=diag)

    # 3) mandatory rerank (fail closed)
    timer = time.monotonic()
    documents = [
        RerankDocument(id=str(point.id), text=str((point.payload or {})["text"]))
        for point in candidates
    ]
    try:
        rerank_response = await local_models.rerank(
            RerankRequest(query=question, documents=documents)
        )
    except LocalModelsError as exc:
        raise RetrievalError(
            ERROR_RERANKER_UNAVAILABLE,
            f"mandatory reranker unavailable: {exc}",
        ) from exc
    diag.model_ids["reranker"] = rerank_response.model_id
    scores = {entry.id: entry.score for entry in rerank_response.scores}
    # descending score, tie-break by ascending id (stable, documented)
    ordered = sorted(candidates, key=lambda p: (-scores[str(p.id)], str(p.id)))
    diag.reranked_count = len(ordered)
    diag.rerank_scores = {str(p.id): scores[str(p.id)] for p in ordered}
    diag.stage_ms["rerank"] = int((time.monotonic() - timer) * 1000)

    # 4) context selection under the LLM budget with reasoning reserve
    timer = time.monotonic()
    question_tokens = length_fn(question)
    budget = effective_context_budget(settings, question_tokens)
    context: list[ContextChunk] = []
    used_tokens = 0
    for point in ordered:
        if len(context) >= settings.context_top_k:
            break
        payload = point.payload or {}
        chunk_tokens = length_fn(str(payload["text"]))
        if used_tokens + chunk_tokens > budget:
            diag.dropped_by_budget_chunk_ids.append(str(point.id))
            continue  # whole-chunk selection only; no partial truncation
        used_tokens += chunk_tokens
        context.append(
            ContextChunk(
                chunk_id=str(point.id),
                chunk_index=int(payload["chunk_index"]),
                text=str(payload["text"]),
                section=payload.get("section"),
                page_from=int(payload["page_from"]),
                page_to=int(payload["page_to"]),
                char_start=int(payload["char_start"]),
                char_end=int(payload["char_end"]),
                rerank_score=scores[str(point.id)],
                segments=list(payload.get("segments") or []),
            )
        )
    diag.context_count = len(context)
    diag.context_chunk_ids = [chunk.chunk_id for chunk in context]
    diag.stage_ms["context"] = int((time.monotonic() - timer) * 1000)

    # 5) diagnostics replay (report-only; ADR 004)
    if debug:
        await _replay_branch_counts(
            qdrant, settings, flt, dense_query, sparse_query, diag
        )

    return RetrievalResult(
        context=context,
        allowed_chunk_ids=diag.context_chunk_ids,
        diagnostics=diag,
    )


async def _replay_branch_counts(
    qdrant: AsyncQdrantClient,
    settings: Settings,
    flt: models.Filter,
    dense_query: list[float],
    sparse_query: Any,
    diag: Diagnostics,
) -> None:
    try:
        responses = await qdrant.query_batch_points(
            collection_name=settings.qdrant_collection,
            requests=[
                models.QueryRequest(
                    query=dense_query, using=DENSE_NAME, filter=flt,
                    limit=settings.dense_top_k, with_payload=False,
                ),
                models.QueryRequest(
                    query=models.SparseVector(
                        indices=sparse_query.indices, values=sparse_query.values
                    ),
                    using=SPARSE_NAME, filter=flt,
                    limit=settings.sparse_top_k, with_payload=False,
                ),
            ],
        )
        diag.dense_returned = len(responses[0].points)
        diag.sparse_returned = len(responses[1].points)
        diag.branch_counts_source = "diagnostic_replay"
    except Exception as exc:
        diag.dense_returned = None
        diag.sparse_returned = None
        diag.branch_counts_source = None
        diag.diagnostic_warnings.append(
            f"branch count replay failed: {type(exc).__name__}"
        )


def _freeze_candidates(
    freeze_dir: Path,
    question: str,
    scope: RetrievalScope,
    candidates: list[Any],
) -> None:
    """Persist fused candidates for honest encoder/reranker comparison (stage 9)."""
    try:
        freeze_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            f"{scope.doc_id}|{scope.accepted_run_id}|{question}".encode()
        ).hexdigest()[:16]
        payload = {
            "question": question,
            "doc_id": scope.doc_id,
            "accepted_run_id": scope.accepted_run_id,
            "index_fingerprint": scope.index_fingerprint,
            "candidates": [
                {
                    "chunk_id": str(point.id),
                    "chunk_index": (point.payload or {}).get("chunk_index"),
                    "fused_rank": rank,
                    "fused_score": float(point.score) if point.score is not None else None,
                }
                for rank, point in enumerate(candidates)
            ],
        }
        (freeze_dir / f"candidates-{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("freeze_candidates_failed", error=type(exc).__name__)
