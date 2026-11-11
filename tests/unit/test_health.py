import httpx
from fastapi import FastAPI


class FakeRepo:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    async def ping(self) -> bool:
        if not self._ok:
            raise ConnectionError("down")
        return True


async def test_live_returns_200(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_ready_is_honest_about_missing_models(
    client: httpx.AsyncClient, test_app: FastAPI
) -> None:
    test_app.state.redis_repo = FakeRepo(ok=True)
    test_app.state.qdrant_repo = FakeRepo(ok=True)
    # No local_models attribute -> stage-1 honest state.
    if hasattr(test_app.state, "local_models"):
        del test_app.state.local_models

    response = await client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["components"]["redis"] == "ok"
    assert body["components"]["qdrant"] == "ok"
    assert body["components"]["local_models"] == "not_initialized"
    assert body["model_ready"] is False


async def test_ready_reports_real_dependency_failure(
    client: httpx.AsyncClient, test_app: FastAPI
) -> None:
    test_app.state.redis_repo = FakeRepo(ok=False)
    test_app.state.qdrant_repo = FakeRepo(ok=True)

    response = await client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["components"]["redis"].startswith("unavailable")
    assert body["components"]["qdrant"] == "ok"
