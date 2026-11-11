import httpx
import pytest
from app.config.settings import Settings
from app.main import create_app
from fastapi import FastAPI


@pytest.fixture()
def test_app() -> FastAPI:
    # Lifespan is intentionally NOT run: tests inject fake dependencies into state.
    return create_app(settings=Settings(_env_file=None))  # type: ignore[call-arg]


@pytest.fixture()
async def client(test_app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
