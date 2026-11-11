"""Thin async HTTP client for the local-models service.

API/worker processes use this client and MUST NOT import model weights or
torch-based adapters (single-owner rule, ADR 002/ADR 001; measured in spike S8).
"""

from __future__ import annotations

import httpx

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


class LocalModelsClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        from app.reliability.breaker import get_breaker

        self._breaker = get_breaker("local-models")

    async def health(self) -> ServiceHealth:
        try:
            response = await self._client.get("/v1/health")
            response.raise_for_status()
            return ServiceHealth.model_validate(response.json())
        except httpx.ConnectError:
            return ServiceHealth(status=ServiceStatus.NOT_INITIALIZED)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models health failed: {type(exc).__name__}") from exc

    async def _post_with_policy(self, path: str, payload: dict, *, op: str,
                                attempts: int) -> dict:
        from app.reliability.breaker import CircuitOpenError
        from app.reliability.policy import retry_async

        async def once() -> dict:
            self._breaker.before_call()
            try:
                response = await self._client.post(path, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 502, 503, 504):
                    self._breaker.record_transient_failure()
                else:
                    self._breaker.record_config_failure()
                raise
            except httpx.HTTPError:
                self._breaker.record_transient_failure()
                raise
            self._breaker.record_success()
            return dict(response.json())

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
        return EmbedResponse.model_validate(raw)

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        raw = await self._post_with_policy(
            "/v1/rerank", request.model_dump(mode="json"),
            op="local_models.rerank", attempts=2,
        )
        return RerankResponse.model_validate(raw)

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
            return ServiceHealth.model_validate(response.json())
        except httpx.ConnectError:
            return ServiceHealth(status=ServiceStatus.NOT_INITIALIZED)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models health failed: {type(exc).__name__}") from exc

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        from app.reliability.policy import retry_sync

        def once() -> EmbedResponse:
            response = self._client.post("/v1/embed", json=request.model_dump(mode="json"))
            response.raise_for_status()
            return EmbedResponse.model_validate(response.json())

        try:
            return retry_sync(once, op_name="local_models.embed_sync", max_attempts=3)
        except httpx.HTTPError as exc:
            raise LocalModelsError(f"local-models embed failed: {type(exc).__name__}") from exc

    def close(self) -> None:
        self._client.close()
