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


async def test_ready_requires_the_configured_llm_model(
    client: httpx.AsyncClient, test_app: FastAPI
) -> None:
    from app.inference.contract import ServiceHealth, ServiceStatus

    class Models:
        async def health(self) -> ServiceHealth:
            return ServiceHealth(status=ServiceStatus.OK)

    test_app.state.redis_repo = test_app.state.qdrant_repo = FakeRepo()
    test_app.state.local_models = Models()
    code = 404

    def show(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        assert request.method == "POST"
        return httpx.Response(code, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(show)) as probe:
        test_app.state.health_http = probe
        response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "5"
        assert response.json()["components"]["llm"].startswith("unavailable")
        code = 200
        response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["model_ready"] is True


async def test_offline_start_requires_bm25_assets(tmp_path) -> None:
    import pytest
    from app.config.settings import Settings
    from app.main import create_app

    app = create_app(Settings(
        _env_file=None, offline=True, fastembed_cache_dir=str(tmp_path),
    ))
    with pytest.raises(RuntimeError, match=r"Qdrant/bm25.*offline cache"):
        async with app.router.lifespan_context(app):
            pytest.fail("API accepted requests without BM25 assets")
