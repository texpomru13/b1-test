"""Application settings: the env-var contract from IMPLEMENTATION_PLAN.md §10.3.

Unknown keys in a dotenv file are rejected (extra="forbid"); invalid values fail
fast at startup with a readable per-field report. Secrets are SecretStr and never
appear in dumps or logs.

Every field here has an effect. Most are read by the app processes (api, worker,
dispatcher). A small group is launcher/compose-level: those are consumed by
`./run` (profile resolution) and `compose.yaml` (the local-models service) and
are declared here only so that one `.env` validates for both. They are marked
"launcher/compose-level" below. compose.yaml passes the app-level variables
through to the containers, so `.env` and the shell environment work in the
Docker path as well as on the host.
"""

from __future__ import annotations

import enum
from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(enum.StrEnum):
    AUTO = "auto"
    OLLAMA = "ollama"
    DMR = "dmr"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Device(enum.StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    METAL = "metal"
    NVIDIA = "nvidia"


class FusionMethod(enum.StrEnum):
    RRF = "rrf"
    DBSF = "dbsf"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    # --- Provider ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "auto"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Profile selection (launcher/compose-level: consumed by ./run) ---
    device: Device = Device.AUTO
    quality_profile: str = "quality"
    profile_strict: bool = False
    offline: bool = False

    # --- Local LLM runtime ---
    ollama_base_url: str = "http://ollama:11434"
    # DMR is a LOCAL OpenAI-compatible endpoint (ADR 005): from containers via
    # model-runner.docker.internal, never through a mounted Docker socket.
    dmr_base_url: str = "http://model-runner.docker.internal/engines/v1"
    llm_read_timeout_s: int = Field(default=300, gt=0)
    qa_deadline_s: int = Field(default=900, gt=0)
    local_context_tokens: int = Field(default=16384, gt=0)
    llm_max_output_tokens: int = Field(default=2048, gt=0)
    llm_reasoning_budget_tokens: int = Field(default=4096, ge=0)

    # --- Embeddings ---
    # launcher/compose-level: the encoder is owned by the local-models service.
    # ./run maps EMBEDDING_MODEL to a pinned revision; compose maps
    # EMBEDDING_BATCH_SIZE to the service's encode batch.
    embedding_model: str = "auto"
    embedding_batch_size: int = Field(default=4, gt=0)
    embedding_instruction_version: str = "1"
    # Path to a tokenizer.json of the resolved embedding model (provisioned by
    # models-init in stages 4/8). None -> documented approximate token length.
    embedding_tokenizer_path: str | None = None

    # --- Reranker ---
    # launcher/compose-level: the reranker is owned by the local-models service.
    # ./run maps RERANKER_MODEL to a pinned revision + adapter; compose maps
    # RERANKER_BATCH_SIZE to the service's scoring batch.
    reranker_model: str = "auto"
    reranker_batch_size: int = Field(default=2, gt=0)

    # Optional dir for frozen fused candidates (honest reranker comparison, stage 9).
    freeze_candidates_dir: str | None = None

    # --- Retrieval ---
    dense_top_k: int = Field(default=60, gt=0)
    sparse_top_k: int = Field(default=60, gt=0)
    fusion_method: FusionMethod = FusionMethod.RRF
    rerank_input_k: int = Field(default=60, gt=0)
    context_top_k: int = Field(default=8, gt=0)

    # --- Sparse / BM25 (model id is fixed: app/retrieval/bm25.py) ---
    bm25_language: str = "russian"
    bm25_avg_len: int = Field(default=256, gt=0)
    lexical_normalizer_version: int = Field(default=1, ge=1)
    # Persistent cache dir for fastembed assets (None -> library default in tmp).
    fastembed_cache_dir: str | None = None

    # --- Chunking ---
    chunk_tokens: int = Field(default=450, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    context_budget_tokens: int = Field(default=6000, gt=0)

    # --- Storage ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "document_chunks"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"
    documents_dir: str = "/data/documents"

    # --- Limits ---
    max_file_mb: int = Field(default=25, gt=0)
    max_pages: int = Field(default=100, gt=0)
    # Upper bound is the structural cap of the request schema (app/schemas/qa.py).
    max_question_chars: int = Field(default=4000, gt=0, le=100_000)
    max_concurrent_qa: int = Field(default=1, gt=0)

    # --- Local inference service ---
    local_models_url: str = "http://local-models:8081"
    local_models_timeout_s: int = Field(default=300, gt=0)

    # --- Evaluation ---
    allow_paid_eval: bool = False
    eval_max_cost_usd: float | None = Field(default=None, gt=0.0)

    # --- Grounding (claim verification is mandatory and has no switch) ---
    max_repair_attempts: int = Field(default=1, ge=0)
    max_llm_calls_per_question: int = Field(default=4, gt=0)

    # --- Reliability ---
    qdrant_max_attempts: int = Field(default=3, gt=0)
    llm_max_attempts: int = Field(default=2, gt=0)
    job_max_attempts: int = Field(default=3, gt=0)
    reconcile_interval_s: int = Field(default=30, gt=0)
    failed_source_ttl_s: int = Field(default=7 * 86400, gt=0)

    # --- Privacy ---
    allow_cloud_fallback: bool = False

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Settings:
        problems: list[str] = []
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            problems.append(
                f"CHUNK_OVERLAP_TOKENS ({self.chunk_overlap_tokens}) must be smaller than "
                f"CHUNK_TOKENS ({self.chunk_tokens})"
            )
        if self.context_top_k > self.rerank_input_k:
            problems.append(
                f"CONTEXT_TOP_K ({self.context_top_k}) must not exceed "
                f"RERANK_INPUT_K ({self.rerank_input_k})"
            )
        if self.rerank_input_k > self.dense_top_k + self.sparse_top_k:
            problems.append(
                f"RERANK_INPUT_K ({self.rerank_input_k}) cannot exceed the fused candidate "
                f"maximum DENSE_TOP_K + SPARSE_TOP_K ({self.dense_top_k + self.sparse_top_k})"
            )
        budget_sum = (
            self.context_budget_tokens
            + self.llm_reasoning_budget_tokens
            + self.llm_max_output_tokens
        )
        if budget_sum >= self.local_context_tokens:
            problems.append(
                f"CONTEXT_BUDGET_TOKENS + LLM_REASONING_BUDGET_TOKENS + LLM_MAX_OUTPUT_TOKENS "
                f"({budget_sum}) must stay below LOCAL_CONTEXT_TOKENS "
                f"({self.local_context_tokens}) to leave room for system prompt and question"
            )
        if self.allow_paid_eval and self.eval_max_cost_usd is None:
            problems.append(
                "ALLOW_PAID_EVAL=true requires an explicit EVAL_MAX_COST_USD spend cap"
            )
        if self.allow_cloud_fallback:
            problems.append(
                "ALLOW_CLOUD_FALLBACK must stay false: automatic local->cloud fallback "
                "is forbidden by the plan"
            )
        if problems:
            raise ValueError("invalid configuration: " + "; ".join(problems))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
