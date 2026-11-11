"""Provider factory with per-provider capability maps (plan §3.3, ADR 005).

Adapters: ollama (native API via langchain-ollama), dmr (LOCAL OpenAI-compatible
endpoint — NOT the cloud OpenAI adapter and NOT langchain-ollama), openai
(Responses API, store=false), anthropic (separate system prompt; temperature /
top_p / top_k are NEVER sent — claude-sonnet-5-5 rejects them).

Rules: a configured API key alone NEVER enables cloud; openai/anthropic without a
key is a configuration error; there is no local->cloud fallback of any kind.
The structured-output method differs per provider and lives in the capability map.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import SecretStr

from app.config.settings import LLMProvider, LLMThinking, Settings
from app.observability.logging import get_logger

logger = get_logger("providers.factory")


class ProviderConfigError(Exception):
    pass


class ModelOutputInvalid(Exception):
    """The model produced unparseable/truncated output (502 after one repair)."""


class ProviderUnavailable(Exception):
    """Transport-level failure talking to the model runtime (503).

    `uncertain=True` means the request may have been SENT and processed/billed
    by the provider (read timeout after send) — callers must log it as an
    uncertain outcome, never as a provably unbilled call (§8.6).
    """

    def __init__(self, message: str, *, uncertain: bool = False) -> None:
        super().__init__(message)
        self.uncertain = uncertain


@dataclass(frozen=True)
class CapabilityMap:
    provider: str
    model: str
    schema_method: str  # ollama_format | openai_json_schema | anthropic_json_schema
    supports_reasoning_split: bool
    sampling_params_allowed: bool  # False -> temperature/top_p/top_k never sent
    is_local: bool
    base_url: str | None


class ChatAdapter(Protocol):
    capabilities: CapabilityMap

    async def generate_json(
        self, *, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> tuple[dict, dict[str, int]]:
        """Returns (parsed JSON object, token usage). Raises ModelOutputInvalid /
        ProviderUnavailable."""
        ...


def _parse_json_content(content: str) -> dict:
    text = content.strip()
    if not text:
        raise ModelOutputInvalid("model returned empty content")
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelOutputInvalid(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ModelOutputInvalid("model output JSON is not an object")
    return parsed


def _usage_from_message(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }


class _OpenAICompatibleAdapter:
    """Chat-completions adapter for DMR (local) and cloud OpenAI."""

    def __init__(self, capabilities: CapabilityMap, chat_model: Any) -> None:
        self.capabilities = capabilities
        self._model = chat_model

    async def generate_json(
        self, *, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> tuple[dict, dict[str, int]]:
        bound = self._model.bind(
            response_format={"type": "json_schema", "json_schema": json_schema},
            max_tokens=max_tokens,
        )
        try:
            message = await bound.ainvoke(
                [("system", system), ("user", user)]
            )
        except ModelOutputInvalid:
            raise
        except Exception as exc:  # transport/runtime failure
            name = type(exc).__name__
            raise ProviderUnavailable(
                f"{self.capabilities.provider} chat failed: {name}",
                uncertain="Timeout" in name or "timeout" in str(exc).lower(),
            ) from exc
        content = message.content if isinstance(message.content, str) else str(message.content)
        if not content.strip():
            # qwen thinking may land in reasoning_content with no final answer
            raise ModelOutputInvalid(
                "model returned no final content (possibly truncated reasoning)"
            )
        return _parse_json_content(content), _usage_from_message(message)


class _OllamaAdapter:
    def __init__(self, capabilities: CapabilityMap, chat_model: Any) -> None:
        self.capabilities = capabilities
        self._model = chat_model

    async def generate_json(
        self, *, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> tuple[dict, dict[str, int]]:
        bound = self._model.bind(format=json_schema["schema"], options={"num_predict": max_tokens})
        try:
            message = await bound.ainvoke([("system", system), ("user", user)])
        except Exception as exc:
            raise ProviderUnavailable(
                f"ollama chat failed: {type(exc).__name__}"
            ) from exc
        content = message.content if isinstance(message.content, str) else str(message.content)
        return _parse_json_content(content), _usage_from_message(message)


class _AnthropicAdapter:
    def __init__(self, capabilities: CapabilityMap, chat_model: Any) -> None:
        self.capabilities = capabilities
        self._model = chat_model

    async def generate_json(
        self, *, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> tuple[dict, dict[str, int]]:
        # Anthropic native json_schema output (S4: method='json_schema' ->
        # output_config.format). No temperature/top_p/top_k are ever set.
        structured = self._model.with_structured_output(
            json_schema["schema"], method="json_schema", include_raw=True
        )
        try:
            result = await structured.ainvoke([("system", system), ("user", user)])
        except Exception as exc:
            raise ProviderUnavailable(
                f"anthropic chat failed: {type(exc).__name__}"
            ) from exc
        parsed = result.get("parsed")
        if parsed is None:
            raise ModelOutputInvalid("anthropic returned no parsed structured output")
        return dict(parsed), _usage_from_message(result.get("raw"))


def build_chat_adapter(settings: Settings) -> ChatAdapter:
    provider = settings.llm_provider
    if provider == LLMProvider.AUTO:
        # Resolution (stage 8) only ever picks LOCAL providers; default runtime here:
        provider = LLMProvider.OLLAMA
    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        capabilities = CapabilityMap(
            provider="ollama", model=settings.llm_model,
            schema_method="ollama_format", supports_reasoning_split=False,
            sampling_params_allowed=True, is_local=True,
            base_url=settings.ollama_base_url,
        )
        # num_ctx is set explicitly: Ollama's own default context depends on the
        # host (4096 on small machines, up to the model maximum on large ones), so
        # leaving it unset makes the §6.3 budget (LOCAL_CONTEXT_TOKENS) fiction.
        thinking = {LLMThinking.AUTO: None, LLMThinking.ON: True,
                    LLMThinking.OFF: False}[settings.llm_thinking]
        ollama_model = ChatOllama(
            model=settings.llm_model, base_url=settings.ollama_base_url,
            num_ctx=settings.local_context_tokens, reasoning=thinking,
        )
        return _OllamaAdapter(capabilities, ollama_model)
    if provider == LLMProvider.DMR:
        from langchain_openai import ChatOpenAI

        capabilities = CapabilityMap(
            provider="dmr", model=settings.llm_model,
            schema_method="openai_json_schema", supports_reasoning_split=False,
            sampling_params_allowed=True, is_local=True,
            base_url=settings.dmr_base_url,
        )
        dmr_model = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.dmr_base_url,
            api_key=SecretStr("dmr-local-no-key-required"),  # local; never a cloud key
            use_responses_api=False,  # DMR is chat-completions only (S9)
            timeout=settings.llm_read_timeout_s,
            max_retries=0,  # retry ownership: stage-7 policy layer, not the SDK
        )
        return _OpenAICompatibleAdapter(capabilities, dmr_model)
    if provider == LLMProvider.OPENAI:
        if settings.openai_api_key is None:
            raise ProviderConfigError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY (a key alone never "
                "enables cloud; the provider must be chosen explicitly)"
            )
        from langchain_openai import ChatOpenAI

        capabilities = CapabilityMap(
            provider="openai", model=settings.llm_model,
            schema_method="openai_json_schema", supports_reasoning_split=True,
            sampling_params_allowed=True, is_local=False, base_url=None,
        )
        openai_model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            use_responses_api=True,
            store=False,  # no provider-side persistence of our requests
            timeout=settings.llm_read_timeout_s,
            max_retries=0,
        )
        return _OpenAICompatibleAdapter(capabilities, openai_model)
    if provider == LLMProvider.ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ProviderConfigError(
                "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY"
            )
        from langchain_anthropic import ChatAnthropic

        capabilities = CapabilityMap(
            provider="anthropic", model=settings.llm_model,
            schema_method="anthropic_json_schema", supports_reasoning_split=True,
            sampling_params_allowed=False,  # sonnet-5-5 rejects sampling params
            is_local=False, base_url=None,
        )
        # NOTE: temperature/top_p/top_k intentionally absent (default None -> omitted).
        anthropic_model = ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_read_timeout_s,
            max_retries=0,
        )
        return _AnthropicAdapter(capabilities, anthropic_model)
    raise ProviderConfigError(f"unknown provider '{provider}'")
