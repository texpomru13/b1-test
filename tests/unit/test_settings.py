from pathlib import Path

import pytest
from app.config.settings import Settings
from pydantic import ValidationError


def make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_valid() -> None:
    settings = make()
    assert settings.llm_provider.value == "ollama"
    assert settings.dense_top_k == 60
    assert settings.context_top_k == 8
    assert settings.allow_paid_eval is False


def test_unknown_dotenv_key_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TOTALLY_UNKNOWN_SETTING=1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"TOTALLY_UNKNOWN_SETTING|extra"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        make(chunk_tokens=100, chunk_overlap_tokens=100)


def test_context_top_k_cannot_exceed_rerank_input() -> None:
    with pytest.raises(ValidationError, match="CONTEXT_TOP_K"):
        make(context_top_k=61, rerank_input_k=60)


def test_rerank_input_bounded_by_branch_sum() -> None:
    with pytest.raises(ValidationError, match="RERANK_INPUT_K"):
        make(dense_top_k=10, sparse_top_k=10, rerank_input_k=30, context_top_k=5)


def test_token_budgets_must_fit_local_context() -> None:
    with pytest.raises(ValidationError, match="LOCAL_CONTEXT_TOKENS"):
        make(local_context_tokens=8000, context_budget_tokens=6000)


def test_paid_eval_requires_cost_cap() -> None:
    with pytest.raises(ValidationError, match="EVAL_MAX_COST_USD"):
        make(allow_paid_eval=True)


def test_cloud_fallback_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="ALLOW_CLOUD_FALLBACK"):
        make(allow_cloud_fallback=True)


def test_negative_limit_rejected() -> None:
    with pytest.raises(ValidationError):
        make(max_file_mb=0)


def test_secret_never_in_repr() -> None:
    settings = make(openai_api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(settings)
    assert "sk-super-secret" not in settings.model_dump_json()
