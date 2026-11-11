"""Indexing must survive a temporarily unavailable vector store: bounded retries
for every Qdrant operation, and a clear QDRANT_UNAVAILABLE job error (never a
silent `running` state) once the bound is exhausted."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from app.config.settings import Settings
from app.inference.contract import EmbedResponse
from app.jobs.indexer import IndexingError, run_indexing
from qdrant_client.http.exceptions import ResponseHandlingException


def _connect_error() -> Exception:
    return ResponseHandlingException(httpx.ConnectError("connection refused"))


class FakeLifecycle:
    def set_fields(self, job_id: str, fields: dict[str, str]) -> None:
        pass

    def renew_lease(self, job_id: str, token: str, *, ttl_ms: int) -> bool:
        return True


class FakeEmbedder:
    def embed(self, request: Any) -> EmbedResponse:
        return EmbedResponse(
            vectors=[[0.1, 0.2] for _ in request.texts], model_id="enc",
            revision="r", dim=2,
        )


class FakeBm25:
    def embed_documents(self, texts: list[str]) -> list[Any]:
        return [object() for _ in texts]


class FakeIndexRepo:
    def __init__(self, *, ensure_failures: int = 0, upsert_failures: int = 0) -> None:
        self.ensure_failures = ensure_failures
        self.upsert_failures = upsert_failures
        self.ensure_calls = 0
        self.upsert_calls = 0
        self.stored = 0

    def ensure_collection(self, *, dense_dim: int, computed_fingerprint: str) -> None:
        self.ensure_calls += 1
        if self.ensure_calls <= self.ensure_failures:
            raise _connect_error()

    def upsert_chunk_batch(self, *, doc_id: str, run_id: str, batch: list[Any],
                           dense_vectors: Any, sparse_vectors: Any) -> None:
        self.upsert_calls += 1
        if self.upsert_calls <= self.upsert_failures:
            raise _connect_error()
        self.stored += len(batch)

    def count_run(self, doc_id: str, run_id: str) -> int:
        return self.stored

    def mark_run_complete_on_head(self, doc_id: str, run_id: str) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr("app.reliability.policy.backoff_delay", lambda *a, **k: 0.0)
    parse_dir = tmp_path / "doc-1" / "parse"
    parse_dir.mkdir(parents=True)
    chunks = [
        {"index": i, "text": f"chunk {i}", "page_from": 1, "page_to": 1,
         "section": None, "char_start": i * 10, "char_end": i * 10 + 7}
        for i in range(3)
    ]
    (parse_dir / "parsed.json").write_text(json.dumps({
        "chunks": chunks, "chunk_tokens": 450, "overlap_tokens": 64,
        "length_fn": "approx", "page_count": 1,
    }), encoding="utf-8")
    return Settings(_env_file=None, documents_dir=str(tmp_path),  # type: ignore[call-arg]
                    qdrant_max_attempts=3)


def _run(repo: FakeIndexRepo, settings: Settings) -> Any:
    return run_indexing(
        job={"doc_id": "doc-1", "title": "t"}, job_id="job-1",
        lifecycle=FakeLifecycle(), lease_token="tok",  # type: ignore[arg-type]
        index_repo=repo, embedder=FakeEmbedder(), bm25=FakeBm25(),  # type: ignore[arg-type]
        settings=settings,
    )


def test_ensure_collection_retried_on_transient_outage(settings: Settings) -> None:
    repo = FakeIndexRepo(ensure_failures=1)
    outcome = _run(repo, settings)
    assert repo.ensure_calls == 2
    assert outcome.chunk_count == 3


def test_upsert_retried_on_transient_outage(settings: Settings) -> None:
    repo = FakeIndexRepo(upsert_failures=2)
    outcome = _run(repo, settings)
    assert repo.upsert_calls == 3  # two failed attempts + the successful one
    assert outcome.chunk_count == 3 and repo.stored == 3


def test_qdrant_outage_before_any_write_is_a_clear_job_error(settings: Settings) -> None:
    repo = FakeIndexRepo(ensure_failures=99)
    with pytest.raises(IndexingError) as excinfo:
        _run(repo, settings)
    assert excinfo.value.code == "QDRANT_UNAVAILABLE"
    assert excinfo.value.retryable is True
    assert excinfo.value.run_id is None  # nothing written -> nothing to compensate
    assert repo.ensure_calls == 3  # exactly QDRANT_MAX_ATTEMPTS


def test_upsert_outage_is_bounded_and_carries_run_for_compensation(
    settings: Settings,
) -> None:
    repo = FakeIndexRepo(upsert_failures=99)
    with pytest.raises(IndexingError) as excinfo:
        _run(repo, settings)
    assert excinfo.value.code == "QDRANT_UNAVAILABLE"
    assert excinfo.value.run_id  # the partial run must be removed by the saga
    assert repo.upsert_calls == 3
