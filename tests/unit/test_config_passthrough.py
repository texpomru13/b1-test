"""Configuration contract: every setting has an effect and can be set through the
environment in the Docker path. Guards against drift between Settings,
compose.yaml and .env.example (assignment: model/connection settings, fusion
method, per-branch candidate counts and reranker in/out counts are env-driven).
"""

import re
from pathlib import Path

from app.config.settings import Settings

ROOT = Path(__file__).resolve().parents[2]

# Declared in Settings so that one .env validates for both the app and the
# launcher, but consumed by ./run (profile resolution) or by compose for the
# local-models service — not forwarded to api/worker/dispatcher.
LAUNCHER_LEVEL = {
    "DEVICE", "QUALITY_PROFILE", "PROFILE_STRICT", "OFFLINE",
    "RERANKER_MODEL", "EMBEDDING_BATCH_SIZE", "RERANKER_BATCH_SIZE",
}
# Not operator knobs of the containerized service.
NOT_FORWARDED = {
    "ALLOW_PAID_EVAL", "EVAL_MAX_COST_USD",  # evaluation harness (host runs)
    "ALLOW_CLOUD_FALLBACK",                  # must stay false (validated guard)
    "EMBEDDING_INSTRUCTION_VERSION", "LEXICAL_NORMALIZER_VERSION",  # index identity
    "FREEZE_CANDIDATES_DIR",                 # evaluation-only debugging aid
}


def _settings_keys() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def _app_env_keys() -> set[str]:
    text = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    block = text.split("environment: &app_env", 1)[1].split("\n    volumes:", 1)[0]
    return set(re.findall(r"^ {6}([A-Z][A-Z0-9_]*):", block, flags=re.MULTILINE))


def test_every_setting_reaches_the_containers_or_is_launcher_level() -> None:
    missing = _settings_keys() - _app_env_keys() - LAUNCHER_LEVEL - NOT_FORWARDED
    assert missing == set(), f"settings not forwarded by compose.yaml: {sorted(missing)}"


def test_assignment_search_settings_are_forwarded() -> None:
    required = {"FUSION_METHOD", "DENSE_TOP_K", "SPARSE_TOP_K", "RERANK_INPUT_K",
                "CONTEXT_TOP_K", "QDRANT_URL", "REDIS_URL", "CELERY_BROKER_URL",
                "LLM_PROVIDER", "LLM_MODEL"}
    assert required <= _app_env_keys()


def test_compose_forwards_only_known_settings() -> None:
    unknown = _app_env_keys() - _settings_keys()
    assert unknown == set(), f"compose.yaml forwards unknown settings: {sorted(unknown)}"


def test_env_example_lists_only_known_settings() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    keys = set(re.findall(r"^#([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE))
    assert keys, ".env.example must list the supported keys"
    unknown = keys - _settings_keys()
    assert unknown == set(), f".env.example lists keys Settings rejects: {sorted(unknown)}"


def test_launcher_reads_the_launcher_level_model_settings() -> None:
    launcher = (ROOT / "run").read_text(encoding="utf-8")
    for key in ("LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_MODEL", "RERANKER_MODEL",
                "DEVICE", "QUALITY_PROFILE", "PROFILE_STRICT"):
        assert f"${{{key}:-}}" in launcher, f"./run ignores {key}"
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "${EMBEDDING_BATCH_SIZE:-" in compose and "${RERANKER_BATCH_SIZE:-" in compose
