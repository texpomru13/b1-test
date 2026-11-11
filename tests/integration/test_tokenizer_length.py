"""Exact tokenizer-aware length function against the REAL Giga tokenizer.

Uses the HF cache left by stage-0 spikes; skipped when the asset is absent.
In deployment the tokenizer.json is provisioned by models-init (stages 4/8) and
wired via EMBEDDING_TOKENIZER_PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.ingestion.chunker import ApproxTokenLength, HFTokenizerLength

_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _find_tokenizer() -> Path | None:
    for repo in ("models--ai-sage--Giga-Embeddings-instruct-480M-0826",
                 "models--ai-sage--Giga-Embeddings-instruct"):
        candidates = sorted((_CACHE / repo).glob("snapshots/*/tokenizer.json"))
        if candidates:
            return candidates[0]
    return None


TOKENIZER = _find_tokenizer()

pytestmark = pytest.mark.skipif(
    TOKENIZER is None, reason="Giga tokenizer.json not present in the local HF cache"
)


def test_exact_length_on_russian_contract_text() -> None:
    assert TOKENIZER is not None
    exact = HFTokenizerLength(TOKENIZER)
    text = (
        "4.2.1 За нарушение сроков поставки Поставщик уплачивает Покупателю "
        "неустойку в размере 0,1% от стоимости товара за каждый день просрочки, "
        "но не более 10% цены договора."
    )
    exact_tokens = exact(text)
    approx_tokens = ApproxTokenLength()(text)
    assert exact_tokens > 0
    assert exact.id.startswith("hf:")
    # The documented approximation must stay within 2x of the real count both
    # ways for this domain text — otherwise the fallback is misconfigured.
    assert approx_tokens <= exact_tokens * 2
    assert exact_tokens <= approx_tokens * 2
