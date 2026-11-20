from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
from redis.exceptions import ConnectionError as RedisConnectionError


async def test_unknown_route_uses_error_envelope(client: httpx.AsyncClient) -> None:
    response = await client.get("/no/such/route")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["retryable"] is False
    assert error["request_id"]


async def test_request_id_header_roundtrip(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live", headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"

    response = await client.get("/health/live")
    assert len(response.headers["X-Request-ID"]) == 32  # generated uuid4 hex


async def test_registry_outage_is_retryable(test_app: FastAPI, client: httpx.AsyncClient) -> None:
    test_app.state.redis_repo = AsyncMock()
    test_app.state.redis_repo.get_job.side_effect = RedisConnectionError()
    response = await client.get("/api/v1/jobs/example")
    assert response.status_code == 503
    assert int(response.headers["Retry-After"]) > 0
    assert response.json()["error"]["retryable"] is True


async def test_api_error_schema_matches_envelope(client: httpx.AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    upload = schema["paths"]["/api/v1/documents"]["post"]["responses"]
    for status in ("409", "413", "415", "422", "503"):
        assert upload[status]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ErrorEnvelope"
        )
    pdf = schema["paths"]["/api/v1/documents/{doc_id}/normalized"]["get"]["responses"]
    assert set(pdf["200"]["content"]) == {"application/pdf"}
