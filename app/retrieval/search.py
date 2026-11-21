"""Retrieve and rerank evidence within one published document.

One Qdrant query fuses the dense and BM25 branches. The reranker orders the
candidates, then the context limit and token budget determine which chunks the
model can cite. Debug mode measures branch sizes in a separate diagnostic query."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

from app.config.settings import FusionMethod, Settings
from app.domain.index_fingerprint import fingerprint_for_embedding
from app.inference.client import LocalModelsClient, LocalModelsError
from app.inference.contract import (
    EmbedRequest,
    InputType,
    RerankDocument,
    RerankRequest,
)
from app.ingestion.chunker import LengthFn
from app.ingestion.scopes import requested_scope_keys, scope_key
from app.reliability.policy import retry_async
from app.retrieval.articles import article_gaps, article_units
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.day_definition import prioritize_day_definition
from app.retrieval.offer_details import prioritize_source_details
from app.retrieval.tables import attach_price_headings, price_table_gaps
from app.storage.qdrant_repo import DENSE_NAME, SPARSE_NAME, FingerprintMismatchError
from qdrant_client import AsyncQdrantClient, models

# Reserve space for the instructions, response schema and framing of source text.
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
    list_heading_id: str | None = None
    scope_id: str | None = None
    scope_label: str | None = None
    table_heading_span: tuple[int, int] | None = None
    scope_heading_span: tuple[int, int] | None = None


def _context_unit(chunk_id: str, payload: dict[str, Any]) -> list[ContextChunk]:
    """Exact sources evaluated together, ordered from the outermost rule to the item."""
    parts: list[ContextChunk] = []
    for key, span in [
        *((span["chunk_id"], span) for span in payload.get("governing_spans", ())),
        (chunk_id, payload),
    ]:
        parts.append(ContextChunk(
            chunk_id=key, chunk_index=int(span["chunk_index"]), text=span["text"],
            section=span.get("section"), page_from=span["page_from"], page_to=span["page_to"],
            char_start=span["char_start"], char_end=span["char_end"],
            segments=span.get("segments") or [], rerank_score=0,
            list_heading_id=parts[-1].chunk_id if parts else None,
            scope_id=payload.get("scope_id"), scope_label=payload.get("scope_label"),
        ))
    return parts


def _scoped_text(chunk: ContextChunk) -> str:
    return f"{chunk.scope_label}\n{chunk.text}" if chunk.scope_label else chunk.text


@dataclass
class Diagnostics:
    dense_returned: int | None = None
    sparse_returned: int | None = None
    fused_returned: int = 0
    reranked_count: int = 0
    context_count: int = 0
    context_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] | None = None  # attached to verified claims
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
    fingerprint = fingerprint_for_embedding(
        dense_response, settings, length_fn_id=getattr(length_fn, "id", "custom"),
        chunk_tokens=settings.chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens,
    )
    if fingerprint != scope.index_fingerprint:
        raise FingerprintMismatchError(scope.index_fingerprint, fingerprint)
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

    if debug:
        await _replay_branch_counts(qdrant, settings, flt, dense_query, sparse_query, diag)
    if not candidates:
        return RetrievalResult(context=[], allowed_chunk_ids=[], diagnostics=diag)

    # 3) mandatory rerank (fail closed)
    timer = time.monotonic()
    units = {str(p.id): _context_unit(str(p.id), p.payload or {}) for p in candidates}
    structural = dict(units)
    gaps = article_gaps(question, units) | price_table_gaps(question, units)
    if gaps:
        await _bridge_source_gaps(structural, gaps, flt, qdrant, settings, scope, diag)
    eligible = article_units(question, structural)
    attach_price_headings(structural)
    units = {key: structural[key] for key in units}
    documents = [
        RerankDocument(id=key, text="\n".join(_scoped_text(part) for part in unit))
        for key, unit in units.items()
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
    if (len(rerank_response.scores) != len(candidates)
            or set(scores) != {d.id for d in documents}
            or not all(math.isfinite(score) for score in scores.values())):
        raise RetrievalError(ERROR_RERANKER_UNAVAILABLE, "invalid reranker score list")
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
    included: set[str] = set()
    labels = {c.scope_label for unit in units.values() for c in unit if c.scope_label}
    requested = requested_scope_keys(question, labels)
    ordered = prioritize_day_definition(question, ordered)
    ordered = prioritize_source_details(question, ordered, units)
    for point in ordered:
        if len(context) >= settings.context_top_k:
            break
        key = str(point.id)
        if eligible is not None and key not in eligible:
            continue
        unit = units[key]
        label = unit[-1].scope_label
        if requested and label and scope_key(label) not in requested:
            continue
        if not (point.payload or {}).get("governing_complete", True):
            continue
        # Top-k and budget both apply to every distinct source in the unit,
        # including headings outside the fused result. An item never gets in alone.
        new = [chunk for chunk in unit if chunk.chunk_id not in included]
        if len(context) + len(new) > settings.context_top_k:
            continue
        tokens = sum(length_fn(_scoped_text(chunk)) for chunk in new)
        if used_tokens + tokens > budget:
            diag.dropped_by_budget_chunk_ids.append(key)
            continue
        used_tokens += tokens
        context.extend(replace(chunk, rerank_score=scores[key]) for chunk in new)
        included.update(chunk.chunk_id for chunk in new)
    diag.context_count = len(context)
    diag.context_chunk_ids = [chunk.chunk_id for chunk in context]
    diag.stage_ms["context"] = int((time.monotonic() - timer) * 1000)

    return RetrievalResult(
        context=context,
        allowed_chunk_ids=diag.context_chunk_ids,
        diagnostics=diag,
    )


async def _bridge_source_gaps(
    structural: dict[str, list[ContextChunk]], gaps: set[int], flt: models.Filter,
    qdrant: AsyncQdrantClient, settings: Settings, scope: RetrievalScope, diag: Diagnostics,
) -> None:
    # One bounded metadata read. The bridge rows establish continuity; they are not
    # added to the fused/reranked candidates. Missing rows cannot confer a heading.
    positions = sorted(gaps)[:48]
    scoped = models.Filter(must=[flt, models.FieldCondition(
        key="chunk_index", match=models.MatchAny(any=positions),
    )])
    try:
        points, _ = await qdrant.scroll(collection_name=settings.qdrant_collection,
                                        scroll_filter=scoped, limit=len(positions),
                                        with_payload=True, with_vectors=False)
    except Exception as exc:
        diag.diagnostic_warnings.append(f"structure lookup failed: {type(exc).__name__}")
        return
    for p in points:
        data = p.payload or {}
        if (data.get("doc_id") == scope.doc_id
                and data.get("ingestion_run_id") == scope.accepted_run_id
                and data.get("index_fingerprint") == scope.index_fingerprint
                and data.get("chunk_index") in gaps):
            structural[str(p.id)] = _context_unit(str(p.id), data)


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
