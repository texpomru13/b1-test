import httpx


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
