"""Index fingerprint: every input that defines the embedding space / chunk
boundaries / sparse behavior (ADR 002, MODELS_AND_SOURCES §3). Changing any
component invalidates the index; swapping LLM/reranker does NOT appear here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json

from app.config.settings import Settings
from app.inference.contract import EmbedResponse
from app.ingestion.canonical import NORMALIZATION_ID
from app.ingestion.pipeline import PARSER_VERSION
from app.retrieval.lexical import LEXICAL_NORMALIZER_ID


def compute_index_fingerprint(
    *,
    embedding_model_id: str,
    embedding_revision: str,
    embedding_dim: int,
    instruction_version: str,
    chunk_tokens: int,
    overlap_tokens: int,
    length_fn_id: str,
    bm25_language: str,
    bm25_avg_len: int,
    fastembed_version: str,
    ocr_provider: str = "tesseract",
) -> str:
    components = {
        "encoder": embedding_model_id,
        "revision": embedding_revision,
        "dim": embedding_dim,
        "query_instruction": f"v{instruction_version}",
        "doc_formatting": "plain-no-instruction",
        "normalize": "l2",
        "parser": PARSER_VERSION,
        "text_normalization": NORMALIZATION_ID,
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "length_fn": length_fn_id,
        "sparse_model": "Qdrant/bm25",
        "bm25_language": bm25_language,
        "bm25_avg_len": bm25_avg_len,
        "fastembed": fastembed_version,
        "lexical_normalizer": LEXICAL_NORMALIZER_ID,
    }
    if ocr_provider != "tesseract":
        components["ocr_provider"] = ocr_provider
    payload = json.dumps(components, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_for_embedding(
    embedding: EmbedResponse, settings: Settings, *, length_fn_id: str,
    chunk_tokens: int, overlap_tokens: int,
) -> str:
    return compute_index_fingerprint(
        embedding_model_id=embedding.model_id, embedding_revision=embedding.revision,
        embedding_dim=embedding.dim, instruction_version=settings.embedding_instruction_version,
        chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens, length_fn_id=length_fn_id,
        bm25_language=settings.bm25_language, bm25_avg_len=settings.bm25_avg_len,
        fastembed_version=importlib.metadata.version("fastembed"),
        ocr_provider=settings.ocr_provider,
    )
