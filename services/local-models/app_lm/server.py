"""local-models service: the single owner of retrieval model weights.

Implements contract v1 (mirrors app/inference/contract.py):
  GET  /v1/health  -> {status, loaded_model, revision, device, residency, ...}
  POST /v1/embed   -> dense vectors (query/document instruction handling)
  POST /v1/rerank  -> 501 until stage 5

Design (ADR 001/002, spikes S6/S8): one resident encoder; inference runs in a
single worker thread behind a lock (bulkhead) so the event loop never blocks and
two encodes never run concurrently. API/worker processes are HTTP clients only.
"""

from __future__ import annotations

import asyncio
import enum
import os
import threading
from typing import Any

import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

CONTRACT_VERSION = 1

# Query instruction v1 (fingerprint input EMBEDDING_INSTRUCTION_VERSION=1).
# Giga model-card template; documents are encoded WITHOUT instruction.
QUERY_INSTRUCTION_V1 = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)


class ServiceStatus(str, enum.Enum):
    OK = "ok"
    LOADING = "loading"
    NOT_INITIALIZED = "not_initialized"
    ERROR = "error"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    input_type: str  # "query" | "document"
    model_id: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[dict[str, str]] = Field(min_length=1)
    model_id: str | None = None


class ModelHolder:
    def __init__(self) -> None:
        self.status = ServiceStatus.NOT_INITIALIZED
        self.error: str | None = None
        self.model: Any = None
        self.model_id = os.environ.get(
            "LM_MODEL_ID", "ai-sage/Giga-Embeddings-instruct-480M-0826"
        )
        self.revision = os.environ.get("LM_REVISION", "")
        self.device = os.environ.get("LM_DEVICE", "cpu")
        self.batch_size = int(os.environ.get("LM_BATCH_SIZE", "4"))
        self.quality_degraded = os.environ.get("LM_QUALITY_DEGRADED", "") == "1"
        self.degraded_reason = os.environ.get("LM_QUALITY_DEGRADED_REASON", "")
        self.dim: int | None = None
        self.lock = threading.Lock()
        # Reranker (mandatory for QA from stage 5; loaded AFTER the encoder).
        self.reranker_model_id = os.environ.get("LM_RERANKER_MODEL", "")
        self.reranker_revision = os.environ.get("LM_RERANKER_REVISION", "")
        self.reranker_adapter_name = os.environ.get("LM_RERANKER_ADAPTER", "qwen_yes_no")
        self.reranker_batch = int(os.environ.get("LM_RERANKER_BATCH", "4"))
        self.reranker: Any = None
        self.reranker_status = (
            ServiceStatus.NOT_INITIALIZED if self.reranker_model_id else ServiceStatus.OK
        )
        self.reranker_error: str | None = None
        self.reranker_lock = threading.Lock()

    @property
    def overall_status(self) -> ServiceStatus:
        order = {
            ServiceStatus.ERROR: 3, ServiceStatus.NOT_INITIALIZED: 2,
            ServiceStatus.LOADING: 1, ServiceStatus.OK: 0,
        }
        worst = max((self.status, self.reranker_status), key=lambda s: order[s])
        return worst

    def load(self) -> None:
        self.status = ServiceStatus.LOADING
        try:
            from sentence_transformers import SentenceTransformer

            if not self.revision:
                raise RuntimeError("LM_REVISION must be pinned (no mutable revisions)")
            model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,  # revision pinned to the S6-audited commit
                device=self.device,
            )
            probe = model.encode(["проверка размерности"], normalize_embeddings=True)
            self.dim = int(probe.shape[1])
            self.model = model
            self.status = ServiceStatus.OK
        except Exception as exc:  # noqa: BLE001 - surfaced via /v1/health
            self.status = ServiceStatus.ERROR
            self.error = f"{type(exc).__name__}: {exc}"
            return
        # Sequential load: the reranker only after the encoder settled (one copy
        # of each set of weights; shared VM memory budget).
        if self.reranker_model_id:
            self.reranker_status = ServiceStatus.LOADING
            try:
                from app_lm.rerankers import load_reranker

                if not self.reranker_revision:
                    raise RuntimeError("LM_RERANKER_REVISION must be pinned")
                self.reranker = load_reranker(
                    self.reranker_adapter_name,
                    self.reranker_model_id,
                    self.reranker_revision,
                    batch_size=self.reranker_batch,
                )
                self.reranker_status = ServiceStatus.OK
            except Exception as exc:  # noqa: BLE001
                self.reranker_status = ServiceStatus.ERROR
                self.reranker_error = f"{type(exc).__name__}: {exc}"

    def encode(self, texts: list[str], input_type: str) -> list[list[float]]:
        assert self.model is not None
        if input_type == "query":
            texts = [QUERY_INSTRUCTION_V1 + text for text in texts]
        with self.lock:  # single in-flight inference (bulkhead)
            vectors = self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True
            )
        return [[float(value) for value in row] for row in vectors]


holder = ModelHolder()
app = FastAPI(title="local-models", version="0.1.0")


@app.on_event("startup")
def _start_loading() -> None:
    threading.Thread(target=holder.load, daemon=True).start()


@app.get("/v1/health")
def health() -> JSONResponse:
    process = psutil.Process()
    body = {
        "status": holder.overall_status.value,
        "loaded_model": holder.model_id if holder.status == ServiceStatus.OK else None,
        "revision": holder.revision or None,
        "device": holder.device,
        "residency": "resident-both" if holder.reranker_model_id else "resident-single",
        "contract_version": CONTRACT_VERSION,
        "dim": holder.dim,
        "rss_mb": round(process.memory_info().rss / 1e6, 1),
        "quality_degraded": holder.quality_degraded,
        "quality_degraded_reason": holder.degraded_reason or None,
        "error": holder.error,
        "encoder_status": holder.status.value,
        "reranker_status": holder.reranker_status.value,
        "reranker_model": holder.reranker_model_id or None,
        "reranker_revision": holder.reranker_revision or None,
        "reranker_adapter": holder.reranker_adapter_name if holder.reranker_model_id else None,
        "reranker_error": holder.reranker_error,
    }
    return JSONResponse(status_code=200, content=body)


@app.post("/v1/embed")
async def embed(request: EmbedRequest) -> JSONResponse:
    if holder.status != ServiceStatus.OK:
        return JSONResponse(
            status_code=503,
            content={"error": f"model is {holder.status.value}", "detail": holder.error},
        )
    if request.model_id and request.model_id != holder.model_id:
        return JSONResponse(
            status_code=409,
            content={"error": "requested model_id does not match the resolved profile",
                     "resolved": holder.model_id},
        )
    if request.input_type not in ("query", "document"):
        return JSONResponse(status_code=422, content={"error": "invalid input_type"})
    vectors = await asyncio.to_thread(holder.encode, request.texts, request.input_type)
    return JSONResponse(
        status_code=200,
        content={
            "vectors": vectors,
            "model_id": holder.model_id,
            "revision": holder.revision,
            "dim": holder.dim,
            "contract_version": CONTRACT_VERSION,
        },
    )


def _rerank_locked(query: str, documents: list[str]) -> list[float]:
    with holder.reranker_lock:  # single in-flight rerank (bulkhead)
        return holder.reranker.score(query, documents)


@app.post("/v1/rerank")
async def rerank(request: RerankRequest) -> JSONResponse:
    if not holder.reranker_model_id:
        return JSONResponse(
            status_code=501,
            content={"error": "no reranker configured (LM_RERANKER_MODEL is empty)"},
        )
    if holder.reranker_status != ServiceStatus.OK:
        return JSONResponse(
            status_code=503,
            content={"error": f"reranker is {holder.reranker_status.value}",
                     "detail": holder.reranker_error},
        )
    if request.model_id and request.model_id != holder.reranker_model_id:
        return JSONResponse(
            status_code=409,
            content={"error": "requested reranker model_id does not match the profile",
                     "resolved": holder.reranker_model_id},
        )
    texts = [doc["text"] for doc in request.documents]
    scores = await asyncio.to_thread(_rerank_locked, request.query, texts)
    return JSONResponse(
        status_code=200,
        content={
            "scores": [
                {"id": doc["id"], "score": score}
                for doc, score in zip(request.documents, scores)
            ],
            "model_id": holder.reranker_model_id,
            "revision": holder.reranker_revision,
            "contract_version": CONTRACT_VERSION,
        },
    )
