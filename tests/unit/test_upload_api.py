from pathlib import Path
from typing import Any

import httpx
import pytest
from app.storage.file_store import FileStore
from app.storage.redis_repo import SubmitResult
from fastapi import FastAPI

PDF_BYTES = b"%PDF-1.4\nminimal\n%%EOF\n"
PDF_PART = ("contract.pdf", PDF_BYTES, "application/pdf")


class FakeRepo:
    def __init__(self) -> None:
        self.verdict = "CREATED"
        self.fail_with: Exception | None = None
        self.jobs: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, str] = {}
        self.submits: list[dict[str, Any]] = []

    async def submit(self, **kwargs: Any) -> SubmitResult:
        if self.fail_with is not None:
            raise self.fail_with
        self.submits.append(kwargs)
        if self.verdict == "CREATED":
            job_id, doc_id = kwargs["job_id"], kwargs["doc_id"]
            self.jobs[job_id] = {
                "job_id": job_id, "doc_id": doc_id, "status": "accepted",
                "stage": "upload", "attempt": "0",
                "created_at": "t0", "updated_at": "t0", **kwargs["job_fields"],
            }
            self.docs[doc_id] = job_id
            return SubmitResult("CREATED", job_id, doc_id)
        return SubmitResult(self.verdict, "prior-job", "prior-doc")

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    async def get_job_id_for_doc(self, doc_id: str) -> str | None:
        return self.docs.get(doc_id)

    async def list_jobs(self, *, offset: int, limit: int) -> tuple[list[dict[str, Any]], bool]:
        items = list(self.jobs.values())[offset : offset + limit]
        return items, len(self.jobs) > offset + limit


@pytest.fixture()
def fake_repo(test_app: FastAPI, tmp_path: Path) -> FakeRepo:
    repo = FakeRepo()
    test_app.state.redis_repo = repo
    test_app.state.file_store = FileStore(tmp_path, max_bytes=1024 * 1024)
    test_app.state._store_root = tmp_path  # type: ignore[attr-defined]
    return repo


def doc_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != ".incoming")


async def test_upload_created_202(client: httpx.AsyncClient, fake_repo: FakeRepo,
                                  test_app: FastAPI) -> None:
    response = await client.post(
        "/api/v1/documents", files={"file": PDF_PART}, data={"title": "Договор"},
        headers={"Idempotency-Key": "k1"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "accepted"
    assert response.headers["Location"] == body["status_url"]
    assert body["status_url"].endswith(body["job_id"])
    # file landed under the server-controlled path for the new doc
    root: Path = test_app.state._store_root
    assert doc_dirs(root) == [body["doc_id"]]
    # fingerprint was derived and passed through
    assert fake_repo.submits[0]["idem_key"] == "k1"


async def test_upload_duplicate_returns_prior_ids_and_drops_new_file(
    client: httpx.AsyncClient, fake_repo: FakeRepo, test_app: FastAPI
) -> None:
    fake_repo.verdict = "DUPLICATE"
    response = await client.post(
        "/api/v1/documents", files={"file": PDF_PART}, headers={"Idempotency-Key": "k1"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["doc_id"] == "prior-doc"
    assert body["job_id"] == "prior-job"
    assert doc_dirs(test_app.state._store_root) == []  # new copy removed


async def test_upload_conflict_409(client: httpx.AsyncClient, fake_repo: FakeRepo,
                                   test_app: FastAPI) -> None:
    fake_repo.verdict = "CONFLICT"
    response = await client.post(
        "/api/v1/documents", files={"file": PDF_PART}, headers={"Idempotency-Key": "k1"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"
    assert doc_dirs(test_app.state._store_root) == []


async def test_redis_down_gives_503_and_file_waits_for_janitor(
    client: httpx.AsyncClient, fake_repo: FakeRepo, test_app: FastAPI
) -> None:
    fake_repo.fail_with = ConnectionError("redis down")
    response = await client.post("/api/v1/documents", files={"file": PDF_PART})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "DEPENDENCY_UNAVAILABLE"
    assert error["retryable"] is True
    root: Path = test_app.state._store_root
    assert len(doc_dirs(root)) == 1  # orphan kept for the janitor (grace period)
    store: FileStore = test_app.state.file_store
    removed = store.cleanup_orphans(is_referenced=lambda _: False, grace_s=0)
    assert len(removed) == 1
    assert doc_dirs(root) == []


async def test_upload_too_large_413(client: httpx.AsyncClient, fake_repo: FakeRepo,
                                    test_app: FastAPI) -> None:
    test_app.state.file_store = FileStore(test_app.state._store_root, max_bytes=5)
    response = await client.post("/api/v1/documents", files={"file": PDF_PART})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_upload_bad_signature_415(client: httpx.AsyncClient, fake_repo: FakeRepo) -> None:
    response = await client.post(
        "/api/v1/documents",
        files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


async def test_job_status_and_404(client: httpx.AsyncClient, fake_repo: FakeRepo) -> None:
    created = await client.post("/api/v1/documents", files={"file": PDF_PART})
    job_id = created.json()["job_id"]

    ok = await client.get(f"/api/v1/jobs/{job_id}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "accepted"
    assert body["error"] is None
    assert body["attempt"] == 0

    missing = await client.get("/api/v1/jobs/nope")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


async def test_documents_list_and_detail(client: httpx.AsyncClient, fake_repo: FakeRepo) -> None:
    created = await client.post(
        "/api/v1/documents", files={"file": PDF_PART}, data={"title": "Поставка"}
    )
    doc_id = created.json()["doc_id"]

    listing = await client.get("/api/v1/documents")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert len(items) == 1
    assert items[0]["doc_id"] == doc_id
    assert items[0]["ready"] is False

    detail = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail.status_code == 200
    assert detail.json()["title"] == "Поставка"
    assert detail.json()["ready"] is False

    missing = await client.get("/api/v1/documents/unknown")
    assert missing.status_code == 404
