import pytest
from app.config.settings import Settings
from app.providers.factory import ProviderConfigError, build_chat_adapter
from app.schemas.qa import AnswerResponse, Citation, QuestionRequest
from pydantic import ValidationError


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --- factory rules: no hidden cloud, keys required explicitly ---


def test_local_providers_need_no_keys() -> None:
    ollama = build_chat_adapter(make_settings(llm_provider="ollama", llm_model="qwen3.6:27b"))
    assert ollama.capabilities.is_local and ollama.capabilities.provider == "ollama"
    dmr = build_chat_adapter(make_settings(llm_provider="dmr", llm_model="ai/qwen3.6:27B"))
    assert dmr.capabilities.is_local and dmr.capabilities.provider == "dmr"
    assert "model-runner.docker.internal" in (dmr.capabilities.base_url or "")


@pytest.mark.parametrize(("mode", "expected"), [("auto", None), ("on", True), ("off", False)])
def test_ollama_context_and_thinking_are_explicit(mode: str, expected: bool | None) -> None:
    adapter = build_chat_adapter(make_settings(
        llm_provider="ollama", llm_model="qwen3.5:9b-q4_K_M",
        local_context_tokens=16384, llm_thinking=mode,
    ))
    model = adapter._model  # type: ignore[attr-defined]
    # num_ctx must match the §6.3 budget instead of Ollama's host-dependent default
    assert model.num_ctx == 16384
    assert model.reasoning is expected


def test_unknown_thinking_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_thinking="sometimes")


def test_cloud_without_key_is_config_error_not_fallback() -> None:
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_chat_adapter(make_settings(llm_provider="openai", llm_model="gpt-6.1-sol"))
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        build_chat_adapter(
            make_settings(llm_provider="anthropic", llm_model="claude-sonnet-5-5")
        )


def test_key_alone_does_not_enable_cloud() -> None:
    adapter = build_chat_adapter(make_settings(
        llm_provider="ollama", llm_model="m", openai_api_key="sk-present-but-ignored",
    ))
    assert adapter.capabilities.provider == "ollama" and adapter.capabilities.is_local


def test_anthropic_never_sends_sampling_params() -> None:
    adapter = build_chat_adapter(make_settings(
        llm_provider="anthropic", llm_model="claude-sonnet-5-5",
        anthropic_api_key="key",
    ))
    assert adapter.capabilities.sampling_params_allowed is False
    payload = adapter._model._get_request_payload(
        [("system", "s"), ("user", "u")]
    )
    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in payload, f"{forbidden} must not be sent"
    assert "extra_body" not in payload or not payload.get("extra_body")
    assert payload["system"] == "s"  # separate system prompt (Messages API)


def test_openai_uses_responses_api_with_store_false() -> None:
    adapter = build_chat_adapter(make_settings(
        llm_provider="openai", llm_model="gpt-6.1-sol", openai_api_key="key",
    ))
    model = adapter._model
    assert model.use_responses_api is True
    assert model.store is False


# --- request/response invariants ---


def test_question_request_rejects_memory_fields() -> None:
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate({"question": "q", "history": []})
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate({"question": "q", "thread_id": "t"})
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate({"question": "q", "messages": []})
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate({"question": "   "})
    ok = QuestionRequest.model_validate({"question": "  Вопрос?  ", "debug": True})
    assert ok.question == "Вопрос?"


def _citation(cid: int = 1) -> Citation:
    return Citation(id=cid, text="цитата", page=2, page_to=2, section=None,
                    chunk_id="pt-1", char_start=10, char_end=20)


def test_answer_invariants() -> None:
    with pytest.raises(ValidationError, match="non-empty citations"):
        AnswerResponse(answer="Ответ [1]", found=True, citations=[])
    with pytest.raises(ValidationError, match="empty citations"):
        AnswerResponse(answer="нет", found=False, citations=[_citation()])
    with pytest.raises(ValidationError, match="unknown citations"):
        AnswerResponse(answer="Ответ [2]", found=True, citations=[_citation(1)])
    with pytest.raises(ValidationError, match="reference citations"):
        AnswerResponse(answer="Ответ без маркеров", found=True, citations=[_citation()])
    with pytest.raises(ValidationError):
        Citation(id=1, text="t", page=0, page_to=1, section=None, chunk_id="c",
                 char_start=0, char_end=1)
    with pytest.raises(ValidationError):
        Citation(id=1, text="t", page=1, page_to=1, section=None, chunk_id="c",
                 char_start=5, char_end=5)
    ok = AnswerResponse(answer="Неустойка 0,1% [1].", found=True,
                        citations=[_citation()])
    assert ok.found and ok.diagnostics is None
