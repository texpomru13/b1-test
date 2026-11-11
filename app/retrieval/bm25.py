"""FastEmbed BM25 sparse adapter (Qdrant/bm25, Russian).

Documents use `embed` (tf-saturated weights), queries use `query_embed`
(deduplicated unit weights) — the asymmetry verified in spike S3. The lexical
preprocessing (lexical.py) is applied symmetrically to both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retrieval.lexical import prepare_bm25_input

BM25_MODEL_ID = "Qdrant/bm25"


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class Bm25Adapter:
    def __init__(self, *, language: str, avg_len: float,
                 cache_dir: str | None = None) -> None:
        from fastembed import SparseTextEmbedding

        if cache_dir:
            self._model = SparseTextEmbedding(
                model_name=BM25_MODEL_ID, language=language, avg_len=avg_len,
                cache_dir=cache_dir,
            )
        else:
            self._model = SparseTextEmbedding(
                model_name=BM25_MODEL_ID, language=language, avg_len=avg_len
            )
        self.language = language
        self.avg_len = avg_len

    @staticmethod
    def _convert(raw: Any) -> SparseVector:
        return SparseVector(
            indices=[int(i) for i in raw.indices.tolist()],
            values=[float(v) for v in raw.values.tolist()],
        )

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        prepared = [prepare_bm25_input(text) for text in texts]
        return [self._convert(raw) for raw in self._model.embed(prepared)]

    def embed_query(self, text: str) -> SparseVector:
        prepared = prepare_bm25_input(text)
        return self._convert(next(iter(self._model.query_embed(prepared))))
