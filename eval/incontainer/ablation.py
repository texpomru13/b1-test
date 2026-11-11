"""Offline retrieval ablation (runs INSIDE the api container; report-only).

Input: env EVAL_QUERIES_JSON = [{"req_id", "doc_id", "question"}...]
Output (stdout JSON): {req_id: {method: [chunk_id, ...] ranked}} for methods
  dense      — dense branch only (top DENSE_TOP_K)
  sparse     — BM25 branch only (top SPARSE_TOP_K)
  hybrid_rrf — the production fused query (one query_points, RRF), top RERANK_INPUT_K
  hybrid_dbsf— same with DBSF fusion
  rrf_rerank — hybrid_rrf candidates re-ordered by the configured reranker
             (local-models /v1/rerank, same tie-break as production)
The scope filter (doc_id + accepted run + fingerprint) is the production one.
Production QA always uses hybrid + mandatory reranker; this is an evaluation-only
comparison of the stages on identical gold spans (plan §11.3).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from app.config.settings import get_settings
from app.inference.client import LocalModelsClient
from app.inference.contract import EmbedRequest, InputType, RerankDocument, RerankRequest
from app.retrieval.bm25 import Bm25Adapter
from app.retrieval.search import RetrievalScope, scope_filter
from app.storage.qdrant_repo import DENSE_NAME, SPARSE_NAME, create_qdrant_client
from app.storage.redis_repo import RedisJobRepository, create_redis_client
from qdrant_client import models


async def main() -> None:
    s = get_settings()
    queries = json.loads(os.environ["EVAL_QUERIES_JSON"])
    repo = RedisJobRepository(create_redis_client(s.redis_url))
    qdrant = create_qdrant_client(s.qdrant_url, timeout_s=60)
    lm = LocalModelsClient(s.local_models_url, timeout_s=600)
    bm25 = Bm25Adapter(language=s.bm25_language, avg_len=float(s.bm25_avg_len),
                       cache_dir=s.fastembed_cache_dir)
    scopes: dict[str, RetrievalScope] = {}
    out: dict[str, dict] = {}
    for q in queries:
        doc_id = q["doc_id"]
        if doc_id not in scopes:
            job = await repo.get_job(await repo.get_job_id_for_doc(doc_id))
            scopes[doc_id] = RetrievalScope(doc_id, job["accepted_run_id"],
                                            job["index_fingerprint"])
        flt = scope_filter(scopes[doc_id])
        dense = (await lm.embed(EmbedRequest(texts=[q["question"]],
                                             input_type=InputType.QUERY))).vectors[0]
        sp = bm25.embed_query(q["question"])
        sparse = models.SparseVector(indices=sp.indices, values=sp.values)
        res: dict[str, list[str]] = {}
        d = await qdrant.query_points(s.qdrant_collection, query=dense, using=DENSE_NAME,
                                      query_filter=flt, limit=s.dense_top_k)
        res["dense"] = [str(p.id) for p in d.points]
        b = await qdrant.query_points(s.qdrant_collection, query=sparse, using=SPARSE_NAME,
                                      query_filter=flt, limit=s.sparse_top_k)
        res["sparse"] = [str(p.id) for p in b.points]
        texts: dict[str, str] = {}
        for name, fusion in (("hybrid_rrf", models.Fusion.RRF),
                             ("hybrid_dbsf", models.Fusion.DBSF)):
            f = await qdrant.query_points(
                s.qdrant_collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE_NAME, filter=flt,
                                    limit=s.dense_top_k),
                    models.Prefetch(query=sparse, using=SPARSE_NAME, filter=flt,
                                    limit=s.sparse_top_k),
                ],
                query=models.FusionQuery(fusion=fusion), query_filter=flt,
                limit=s.rerank_input_k, with_payload=True,
            )
            res[name] = [str(p.id) for p in f.points]
            for p in f.points:
                texts[str(p.id)] = str((p.payload or {})["text"])
        cands = res["hybrid_rrf"]
        if cands:
            rr = await lm.rerank(RerankRequest(
                query=q["question"],
                documents=[RerankDocument(id=c, text=texts[c]) for c in cands]))
            scores = {e.id: e.score for e in rr.scores}
            res["rrf_rerank"] = sorted(cands, key=lambda c: (-scores[c], c))
            res["_rerank_scores"] = {c: scores[c] for c in cands}  # type: ignore[assignment]
            res["_reranker"] = rr.model_id  # type: ignore[assignment]
        else:
            res["rrf_rerank"] = []
        out[q["req_id"]] = res
        print(f"{q['req_id']} done", file=sys.stderr, flush=True)
    await lm.aclose()
    await qdrant.close()
    await repo.aclose()
    json.dump(out, sys.stdout, ensure_ascii=False)


asyncio.run(main())
