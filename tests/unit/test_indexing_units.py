from app.domain.index_fingerprint import compute_index_fingerprint
from app.retrieval.lexical import prepare_bm25_input, synthetic_tokens
from app.storage.qdrant_repo import point_id


def test_synthetic_tokens_port_of_s3() -> None:
    text = "Пункт 4.2.1: неустойка 0,1% по делу № АБ-123/24, сумма 1 000 000 рублей"
    tokens = synthetic_tokens(text.replace("ё", "е"))
    assert "sec4x2x1" in tokens
    assert "pct0x1" in tokens
    assert "refабx123x24" in tokens
    assert "amt1000000" in tokens


def test_bm25_input_symmetric_and_yo_normalized() -> None:
    doc = prepare_bm25_input("Платёж по пункту 4.2.1")
    query = prepare_bm25_input("платёж 4.2.1")
    assert "ё" not in doc and "ё" not in query
    assert "sec4x2x1" in doc and "sec4x2x1" in query  # symmetric synthetic tokens
    # citation text untouched: function returns a NEW string for BM25 only
    assert doc.startswith("Платеж по пункту 4.2.1")


def _fp(**overrides: object) -> str:
    base: dict = dict(
        embedding_model_id="ai-sage/Giga-Embeddings-instruct-480M-0826",
        embedding_revision="1763d603",
        embedding_dim=1024,
        instruction_version="1",
        chunk_tokens=450,
        overlap_tokens=64,
        length_fn_id="hf:tokenizer.json",
        bm25_language="russian",
        bm25_avg_len=256,
        fastembed_version="0.9.1",
    )
    base.update(overrides)
    return compute_index_fingerprint(**base)  # type: ignore[arg-type]


def test_fingerprint_deterministic_and_sensitive() -> None:
    assert _fp() == _fp()
    assert _fp() != _fp(embedding_revision="other")
    assert _fp() != _fp(embedding_dim=2048)
    assert _fp() != _fp(chunk_tokens=600)
    assert _fp() != _fp(length_fn_id="approx-v1:cpt3")
    assert _fp() != _fp(bm25_avg_len=128)
    assert _fp() != _fp(instruction_version="2")


def test_point_id_deterministic() -> None:
    first = point_id("doc", "run", 0)
    assert first == point_id("doc", "run", 0)
    assert first != point_id("doc", "run", 1)
    assert first != point_id("doc", "run2", 0)
