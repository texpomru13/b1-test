"""Liveness and bounded, non-generating dependency checks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any
from urllib.parse import quote

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config.settings import LLMProvider, Settings

router = APIRouter(tags=["health"])
logger = structlog.get_logger("api.health")
_CHECK_TIMEOUT_S = 3.0


async def _checked(name: str, check: Awaitable[str]) -> str:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT_S):
            return await check
    except TimeoutError:
        return "unavailable: timeout"
    except Exception as exc:
        logger.warning("readiness_check_failed", component=name, error=type(exc).__name__)
        return f"unavailable: {type(exc).__name__}"


async def _repo_status(repo: Any) -> str:
    if repo is None or not await repo.ping():
        return "unavailable: not_connected"
    return "ok"


async def _model_status(models: Any) -> str:
    if models is None:
        return "not_initialized"
    return str((await models.health()).status.value)


async def _llm_status(settings: Settings, client: httpx.AsyncClient | None) -> str:
    if client is None:
        return "unavailable: not_connected"
    provider, model = settings.llm_provider, settings.llm_model
    if provider in (LLMProvider.AUTO, LLMProvider.OLLAMA):
        response = await client.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/show", json={"model": model}
        )
    elif provider == LLMProvider.DMR:
        response = await client.get(f"{settings.dmr_base_url.rstrip('/')}/models")
        response.raise_for_status()
        if not any(entry.get("id") == model for entry in response.json()["data"]):
            return "unavailable: model_missing"
    elif provider == LLMProvider.OPENAI:
        if not settings.openai_api_key:
            return "unavailable: missing_key"
        response = await client.get(
            f"https://api.openai.com/v1/models/{quote(model, safe='')}",
            headers={"Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}"},
        )
    else:
        if not settings.anthropic_api_key:
            return "unavailable: missing_key"
        response = await client.get(
            f"https://api.anthropic.com/v1/models/{quote(model, safe='')}",
            headers={"x-api-key": settings.anthropic_api_key.get_secret_value(),
                     "anthropic-version": "2023-06-01"},
        )
    response.raise_for_status()
    return "ok"


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", responses={503: {"description": "Dependency unavailable",
            "headers": {"Retry-After": {"schema": {"type": "integer"}}}}})
async def ready(request: Request) -> JSONResponse:
    state = request.app.state
    checks = {
        "redis": _repo_status(getattr(state, "redis_repo", None)),
        "qdrant": _repo_status(getattr(state, "qdrant_repo", None)),
        "local_models": _model_status(getattr(state, "local_models", None)),
        "llm": _llm_status(state.settings, getattr(state, "health_http", None)),
    }
    statuses = await asyncio.gather(*(_checked(name, check) for name, check in checks.items()))
    components = dict(zip(checks, statuses, strict=True))
    all_ok = all(value == "ok" for value in statuses)
    return JSONResponse(
        status_code=200 if all_ok else 503,
        headers={} if all_ok else {"Retry-After": "5"},
        content={
            "status": "ready" if all_ok else "not_ready",
            "components": components,
            "model_ready": components["local_models"] == components["llm"] == "ok",
        },
    )
