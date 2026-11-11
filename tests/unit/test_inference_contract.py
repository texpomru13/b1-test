from app.inference.contract import (
    CONTRACT_VERSION,
    EmbedRequest,
    EmbedResponse,
    InputType,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankScore,
)


def test_embed_roundtrip() -> None:
    request = EmbedRequest(texts=["пункт 4.2.1"], input_type=InputType.QUERY)
    restored = EmbedRequest.model_validate_json(request.model_dump_json())
    assert restored == request

    response = EmbedResponse(
        vectors=[[0.1, 0.2]], model_id="ai-sage/Giga-Embeddings-instruct",
        revision="2cf0fdc9", dim=2,
    )
    restored_response = EmbedResponse.model_validate_json(response.model_dump_json())
    assert restored_response == response
    assert restored_response.contract_version == CONTRACT_VERSION


def test_rerank_roundtrip() -> None:
    request = RerankRequest(
        query="срок поставки",
        documents=[RerankDocument(id="c1", text="Срок поставки 10 дней.")],
    )
    assert RerankRequest.model_validate_json(request.model_dump_json()) == request

    response = RerankResponse(
        scores=[RerankScore(id="c1", score=6.47)], model_id="Qwen/Qwen3-Reranker-0.6B",
        revision="e61197ed",
    )
    assert RerankResponse.model_validate_json(response.model_dump_json()) == response


def test_empty_texts_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EmbedRequest(texts=[], input_type=InputType.DOCUMENT)
