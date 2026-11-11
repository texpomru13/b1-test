"""Integration smoke against the real compose stack.

Run manually after `./run`: RUN_STACK_TESTS=1 uv run pytest tests/integration -q
Skipped by default so CI and plain pytest runs need no services.
"""

import os

import httpx
import pytest

from tests.helpers.pdfgen import make_corrupt_pdf, make_minimal_docx, make_text_pdf

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_STACK_TESTS") != "1",
    reason="requires the running compose stack (set RUN_STACK_TESTS=1)",
)

BASE_URL = "http://127.0.0.1:8000"


def test_live() -> None:
    response = httpx.get(f"{BASE_URL}/health/live", timeout=5)
    assert response.status_code == 200


def test_ready_becomes_fully_green() -> None:
    import time

    # local-models loads Giga-3B fp32 on CPU: allow minutes on a cold start.
    deadline = time.time() + 600
    body: dict = {}
    status_code = 0
    while time.time() < deadline:
        response = httpx.get(f"{BASE_URL}/health/ready", timeout=10)
        status_code, body = response.status_code, response.json()
        if status_code == 200:
            break
        assert body["components"]["local_models"] in (
            "loading", "not_initialized", "unavailable: ConnectError"
        ), body  # only honest transitional states are acceptable
        time.sleep(5)
    assert status_code == 200, body
    assert body["components"] == {"redis": "ok", "qdrant": "ok", "local_models": "ok"}
    assert body["model_ready"] is True


PDF_BYTES = make_text_pdf(
    ["1. Subject\nThe supplier delivers goods of proper quality to the buyer.",
     "4.2 Penalties\nThe penalty is 0.1 percent per day, capped by total price."]
)
PDF_B_BYTES = make_text_pdf(["Completely different contract text."])


def _wait_terminal(job_url: str, timeout_s: float = 90.0) -> dict:
    import time

    deadline = time.time() + timeout_s
    final: dict = {}
    while time.time() < deadline:
        final = httpx.get(job_url, timeout=5).json()
        if final.get("status") in ("succeeded", "failed"):
            return final
        time.sleep(1)
    return final


def _upload(content: bytes, filename: str, content_type: str, **extra: str) -> httpx.Response:
    return httpx.post(
        f"{BASE_URL}/api/v1/documents",
        files={"file": (filename, content, content_type)},
        data={"title": extra.get("title", "Smoke")},
        headers={"Idempotency-Key": extra["key"]} if "key" in extra else {},
        timeout=60,
    )


def test_parse_errors_have_stable_codes() -> None:
    import uuid

    # corrupt PDF passes the magic check but fails real parsing
    corrupt = _upload(make_corrupt_pdf(), "bad.pdf", "application/pdf",
                      key=f"smoke-corrupt-{uuid.uuid4().hex[:8]}")
    assert corrupt.status_code == 202
    final = _wait_terminal(f"{BASE_URL}{corrupt.json()['status_url']}")
    assert final["status"] == "failed"
    assert final["error"]["code"] == "PARSE_FAILED"

    # 101 pages -> terminal PAGE_LIMIT_EXCEEDED (202 was already returned)
    big = make_text_pdf([f"Page {i} of the oversized contract." for i in range(1, 102)])
    over = _upload(big, "big.pdf", "application/pdf",
                   key=f"smoke-101-{uuid.uuid4().hex[:8]}")
    assert over.status_code == 202
    final_big = _wait_terminal(f"{BASE_URL}{over.json()['status_url']}")
    assert final_big["status"] == "failed"
    assert final_big["error"]["code"] == "PAGE_LIMIT_EXCEEDED"


def test_docx_converts_and_normalized_pdf_served() -> None:
    import uuid

    docx = make_minimal_docx([
        "Договор поставки № АБ-123/24",
        "1. Предмет договора",
        "Поставщик обязуется поставить товар надлежащего качества в срок десять рабочих дней.",
        "4.2 Штрафные санкции",
        "Неустойка составляет 0,1% в день, но не более 10% цены договора.",
    ])
    uploaded = _upload(
        docx, "contract.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"smoke-docx-{uuid.uuid4().hex[:8]}", title="DOCX договор",
    )
    assert uploaded.status_code == 202, uploaded.text
    body = uploaded.json()
    # LibreOffice conversion takes seconds: the first poll proves the upload
    # returned long before the task finished.
    first = httpx.get(f"{BASE_URL}{body['status_url']}", timeout=5).json()
    assert first["status"] in ("accepted", "queued", "running"), first
    final = _wait_terminal(f"{BASE_URL}{body['status_url']}", timeout_s=240)
    # stage 4: DOCX converts, parses AND indexes end-to-end
    assert final["status"] == "succeeded", final
    assert final["error"] is None
    assert final["progress"]["completed"] >= 1

    detail = httpx.get(f"{BASE_URL}{body['document_url']}", timeout=5).json()
    assert detail["page_count"] >= 1
    assert detail["chunk_count"] >= 1

    normalized = httpx.get(f"{BASE_URL}{body['document_url']}/normalized", timeout=10)
    assert normalized.status_code == 200
    assert normalized.headers["content-type"].startswith("application/pdf")
    assert normalized.content.startswith(b"%PDF-")


def test_upload_lifecycle_end_to_end() -> None:
    import time
    import uuid

    key = f"smoke-{uuid.uuid4().hex[:10]}"

    # 1) upload returns 202 BEFORE the task finishes
    created = httpx.post(
        f"{BASE_URL}/api/v1/documents",
        files={"file": ("contract.pdf", PDF_BYTES, "application/pdf")},
        data={"title": "Smoke договор"},
        headers={"Idempotency-Key": key},
        timeout=30,
    )
    assert created.status_code == 202, created.text
    body = created.json()
    job_url = f"{BASE_URL}{body['status_url']}"
    # The 202 body is emitted in the `accepted` state: the API never waits for the
    # task. (A tiny PDF can finish parsing before the first poll, so the
    # first-poll-non-terminal evidence lives in the slower DOCX test below.)
    assert body["status"] == "accepted"

    # 2) same key + same file -> prior IDs
    duplicate = httpx.post(
        f"{BASE_URL}/api/v1/documents",
        files={"file": ("contract.pdf", PDF_BYTES, "application/pdf")},
        data={"title": "Smoke договор"},
        headers={"Idempotency-Key": key},
        timeout=30,
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["doc_id"] == body["doc_id"]
    assert duplicate.json()["job_id"] == body["job_id"]

    # 3) same key + different file -> 409 envelope
    conflict = httpx.post(
        f"{BASE_URL}/api/v1/documents",
        files={"file": ("other.pdf", PDF_B_BYTES, "application/pdf")},
        data={"title": "Smoke договор"},
        headers={"Idempotency-Key": key},
        timeout=30,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "CONFLICT"

    # 4) dispatcher + worker drive the job to the EXPLICIT stage-2 stub terminal state
    deadline = time.time() + 60
    final: dict = {}
    while time.time() < deadline:
        final = httpx.get(job_url, timeout=5).json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    assert final.get("status") == "succeeded", final  # full parse + index + publish
    assert final["error"] is None
    assert final["attempt"] == 1
    assert final["progress"]["completed"] >= 1

    # 5) listing + detail come from our registry, not Celery AsyncResult
    listing = httpx.get(f"{BASE_URL}/api/v1/documents", timeout=5).json()
    assert any(item["doc_id"] == body["doc_id"] for item in listing["items"])
    detail = httpx.get(f"{BASE_URL}{body['document_url']}", timeout=5).json()
    assert detail["ready"] is True  # succeeded via CAS publication
    assert detail["title"] == "Smoke договор"
    assert detail["chunk_count"] >= 1
