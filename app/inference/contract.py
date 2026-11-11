"""local-models service contract (client/server schema), version 1.

Source of truth for the HTTP API between the app (API/worker as clients) and the
single owner of retrieval weights (local-models). The server side is implemented
in stage 4/5; the schemas are shared so both sides pin one contract.

Routes (server): GET /v1/health, POST /v1/embed, POST /v1/rerank.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

CONTRACT_VERSION = 1


class InputType(enum.StrEnum):
    QUERY = "query"
    DOCUMENT = "document"


class ServiceStatus(enum.StrEnum):
    OK = "ok"
    LOADING = "loading"
    NOT_INITIALIZED = "not_initialized"
    ERROR = "error"


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    input_type: InputType
    # None means "the server's resolved profile model"; an explicit id must match it.
    model_id: str | None = None


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    model_id: str
    revision: str
    dim: int
    contract_version: int = CONTRACT_VERSION


class RerankDocument(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[RerankDocument] = Field(min_length=1)
    model_id: str | None = None


class RerankScore(BaseModel):
    id: str
    # Raw model score; scale is model-specific and NOT comparable across models.
    score: float


class RerankResponse(BaseModel):
    scores: list[RerankScore]
    model_id: str
    revision: str
    contract_version: int = CONTRACT_VERSION


class ServiceHealth(BaseModel):
    status: ServiceStatus
    loaded_model: str | None = None
    revision: str | None = None
    device: str | None = None
    residency: str | None = None
    contract_version: int = CONTRACT_VERSION
