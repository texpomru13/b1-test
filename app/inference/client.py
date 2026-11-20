"""Thin async HTTP client for the local-models service.

API/worker processes use this client and MUST NOT import model weights or
torch-based adapters (single-owner rule, ADR 002/ADR 001; measured in spike S8).
"""

from __future__ import annotations

import asyncio

import httpx
from pydantic import BaseModel, ValidationError

from app.inference.contract import (
    EmbedRequest,
    EmbedResponse,
    RerankRequest,
    RerankResponse,
    ServiceHealth,
    ServiceStatus,
)


class LocalModelsError(Exception):
    """Transport or protocol failure talking to local-models."""

def _parse[T: BaseModel](response: httpx.Response, schema: type[T]) -> T:
    try:
        return schema.model_validate_json(response.content)
    except ValidationError as exc:
        raise LocalModelsError(f"invalid local-models {schema.__name__}") from exc


def _vectors(response: httpx.Response, request: EmbedRequest) -> EmbedResponse:
    result = _parse(response, EmbedResponse)
    if len(result.vectors) != len(request.texts) or any(
        len(row) != result.dim for row in result.vectors
    ):
        raise LocalModelsError("local-models returned inconsistent vector dimensions or count")
    return result


class LocalModelsClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        from app.reliability.breaker import get_breaker

        self._breaker = get_breaker("local-models")

    async def health(self) -> ServiceHealth:
        try:
            response = await self._client.get("/v1/health")
            response.raise_for_status()
            return _parse(response, ServiceHealth)
        except httpx.ConnectError:
            return ServiceHealth(status=ServiceStatus.NOT_INITIALIZED)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models health failed: {type(exc).__name__}") from exc

    async def _post_with_policy(self, path: str, payload: dict, *, op: str,
                                attempts: int) -> httpx.Response:
        from app.reliability.breaker import CircuitOpenError
        from app.reliability.policy import classify_exception, retry_async

        async def once() -> httpx.Response:
            self._breaker.before_call()
            try:
                response = await self._client.post(path, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if classify_exception(exc).retry:
                    self._breaker.record_transient_failure()
                else:
                    self._breaker.record_config_failure()
                raise
            except asyncio.CancelledError:
                self._breaker.record_config_failure()
                raise
            except httpx.HTTPError:
                self._breaker.record_transient_failure()
                raise
            self._breaker.record_success()
            return response

        try:
            return await retry_async(once, op_name=op, max_attempts=attempts)
        except CircuitOpenError as exc:
            raise LocalModelsError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"{op} failed: {type(exc).__name__}") from exc

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        raw = await self._post_with_policy(
            "/v1/embed", request.model_dump(mode="json"),
            op="local_models.embed", attempts=3,
        )
        return _vectors(raw, request)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        raw = await self._post_with_policy(
            "/v1/rerank", request.model_dump(mode="json"),
            op="local_models.rerank", attempts=2,
        )
        return _parse(raw, RerankResponse)

    async def aclose(self) -> None:
        await self._client.aclose()


class LocalModelsSyncClient:
    """Sync twin for the Celery worker (same contract, httpx.Client)."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def health(self) -> ServiceHealth:
        try:
            response = self._client.get("/v1/health")
            response.raise_for_status()
            return _parse(response, ServiceHealth)
        except httpx.ConnectError:
            return ServiceHealth(status=ServiceStatus.NOT_INITIALIZED)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models health failed: {type(exc).__name__}") from exc

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        from app.reliability.policy import retry_sync

        def once() -> EmbedResponse:
            response = self._client.post("/v1/embed", json=request.model_dump(mode="json"))
            response.raise_for_status()
            return _vectors(response, request)

        try:
            return retry_sync(once, op_name="local_models.embed_sync", max_attempts=3)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models embed failed: {type(exc).__name__}") from exc

    def close(self) -> None:
        self._client.close()
