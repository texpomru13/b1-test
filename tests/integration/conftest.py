"""Pinned, offline model services shared by the indexing and retrieval tests."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "services" / "local-models"
ENCODER = "ai-sage/Giga-Embeddings-instruct-480M-0826"
ENCODER_REVISION = "1763d603adac8057bd6482a708001b0ed2e0a903"


def _model_service(port: int, reranker: str, revision: str, adapter: str) -> Iterator[str]:
    base_url = f"http://127.0.0.1:{port}"
    url = f"{base_url}/v1/health"
    try:
        current = httpx.get(url, timeout=2).json()
    except httpx.ConnectError:
        current = None
    if current is not None:
        expected = {"status": "ok", "loaded_model": ENCODER, "revision": ENCODER_REVISION,
                    "reranker_model": reranker, "reranker_revision": revision}
        assert expected.items() <= current.items(), current
        yield base_url
        return

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    executable = MODEL_DIR / ".venv" / "bin" / "uvicorn"
    if not executable.exists():
        pytest.skip("run uv sync --project services/local-models to prepare local models")
    for model, pin in [(ENCODER, ENCODER_REVISION), (reranker, revision)]:
        if not (cache / ("models--" + model.replace("/", "--")) / "snapshots" / pin).is_dir():
            pytest.skip(f"offline test needs the cached weights for {model}@{pin}")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "LM_MODEL_ID": ENCODER,
           "LM_REVISION": ENCODER_REVISION, "LM_RERANKER_MODEL": reranker,
           "LM_RERANKER_REVISION": revision, "LM_RERANKER_ADAPTER": adapter}
    process = subprocess.Popen([str(executable), "app_lm.server:app", "--host", "127.0.0.1",
                                "--port", str(port)], cwd=MODEL_DIR, env=env)
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert process.poll() is None, f"{reranker} exited before it was ready"
            try:
                body = httpx.get(url, timeout=3).json()
            except httpx.ConnectError:
                body = {}
            if body.get("status") == "ok":
                break
            assert body.get("status") != "error", body
            time.sleep(1)
        else:
            pytest.fail(f"{reranker} did not become ready")
        yield base_url
    finally:
        process.terminate()
        process.wait(timeout=30)


@pytest.fixture(scope="session")
def bge_models() -> Iterator[str]:
    yield from _model_service(18081, "BAAI/bge-reranker-v2-m3",
                              "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "cross_encoder")


@pytest.fixture(scope="session")
def qwen_models() -> Iterator[str]:
    yield from _model_service(18082, "Qwen/Qwen3-Reranker-0.6B",
                              "e61197ed45024b0ed8a2d74b80b4d909f1255473", "qwen_yes_no")
