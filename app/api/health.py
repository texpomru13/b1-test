"""Liveness and honest readiness.

/health/ready reports the REAL state of every dependency. At stage 1 the
local-models inference service does not exist yet, so readiness honestly stays
not_ready with models="not_initialized" instead of a false healthy.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.inference.contract import ServiceStatus

router = APIRouter(tags=["health"])
logger = structlog.get_logger("api.health")

_CHECK_TIMEOUT_S = 2.0


async def _check(name: str, coro: Any) -> str:
    try:
        async with asyncio.timeout(_CHECK_TIMEOUT_S):
            await coro
        return "ok"
    except TimeoutError:
        return "unavailable: timeout"
    except Exception as exc:
        logger.warning("readiness_check_failed", component=name, error=type(exc).__name__)
        return f"unavailable: {type(exc).__name__}"


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    state = request.app.state
    components: dict[str, str] = {}

    redis_repo = getattr(state, "redis_repo", None)
    components["redis"] = (
        await _check("redis", redis_repo.ping()) if redis_repo else "unavailable: not_connected"
    )

    qdrant_repo = getattr(state, "qdrant_repo", None)
    components["qdrant"] = (
        await _check("qdrant", qdrant_repo.ping())
        if qdrant_repo
        else "unavailable: not_connected"
    )

    local_models = getattr(state, "local_models", None)
    if local_models is None:
        components["local_models"] = ServiceStatus.NOT_INITIALIZED.value
    else:
        try:
            async with asyncio.timeout(_CHECK_TIMEOUT_S):
                health = await local_models.health()
            components["local_models"] = health.status.value
        except Exception as exc:
            logger.warning("readiness_check_failed", component="local_models",
                           error=type(exc).__name__)
            components["local_models"] = f"unavailable: {type(exc).__name__}"

    all_ok = all(value == "ok" for value in components.values())
    body = {
        "status": "ready" if all_ok else "not_ready",
        "components": components,
        "model_ready": components.get("local_models") == "ok",
    }
    return JSONResponse(status_code=200 if all_ok else 503, content=body)
