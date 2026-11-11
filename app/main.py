"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import ValidationError

from app.api.documents import router as documents_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.api.middleware import request_id_middleware
from app.api.questions import router as questions_router
from app.config.settings import Settings, get_settings
from app.inference.client import LocalModelsClient
from app.observability.logging import configure_logging, get_logger
from app.storage.file_store import FileStore
from app.storage.qdrant_repo import QdrantChunkRepository, create_qdrant_client
from app.storage.redis_repo import RedisJobRepository, create_redis_client

logger = get_logger("app.main")


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        print("Invalid configuration (fix environment/.env and restart):", file=sys.stderr)
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"]) or "<settings>"
            print(f"  {location}: {err['msg']}", file=sys.stderr)
        raise SystemExit(3) from exc


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings if settings is not None else _load_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = app_settings
        app.state.redis_repo = RedisJobRepository(create_redis_client(app_settings.redis_url))
        app.state.qdrant_repo = QdrantChunkRepository(
            create_qdrant_client(app_settings.qdrant_url),
            app_settings.qdrant_collection,
        )
        app.state.local_models = LocalModelsClient(
            app_settings.local_models_url, timeout_s=float(app_settings.local_models_timeout_s)
        )
        app.state.file_store = FileStore(
            Path(app_settings.documents_dir),
            max_bytes=app_settings.max_file_mb * 1024 * 1024,
        )
        import asyncio as _asyncio

        app.state.qa_semaphore = _asyncio.Semaphore(app_settings.max_concurrent_qa)
        app.state.qa_runtime = None  # built lazily on the first question
        logger.info(
            "startup_complete",
            llm_provider=app_settings.llm_provider.value,
            qdrant_collection=app_settings.qdrant_collection,
        )
        try:
            yield
        finally:
            await app.state.local_models.aclose()
            await app.state.qdrant_repo.aclose()
            await app.state.redis_repo.aclose()
            logger.info("shutdown_complete")

    app = FastAPI(
        title="Document QA RAG backend",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(questions_router)
    return app


app = create_app()
