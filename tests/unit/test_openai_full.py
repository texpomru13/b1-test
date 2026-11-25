from types import SimpleNamespace

import pytest
from app.inference import openai_server
from app.inference.contract import RerankDocument, RerankRequest
from fastapi import HTTPException


@pytest.fixture
def rerank_input() -> tuple[SimpleNamespace, RerankRequest]:
    state = SimpleNamespace(openai=None, reranker="gpt-5.4-mini")
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    body = RerankRequest(
        query="Какой срок?", documents=[
            RerankDocument(id=str(i), text=f"Пункт {i}.") for i in range(20)
        ],
    )
    return request, body


async def test_openai_reranks_the_head_and_keeps_every_candidate(
    monkeypatch: pytest.MonkeyPatch, rerank_input: tuple[SimpleNamespace, RerankRequest],
) -> None:
    async def post(*_args: object) -> dict:
        return {"status": "completed", "output": [{
            "type": "message", "content": [{"type": "output_text", "text":
                '{"order":["C019","C019","C018","C017","C016","C015","C014","C013","C012"]}'}],
        }]}

    monkeypatch.setattr(openai_server, "provider_post", post)
    request, body = rerank_input
    response = await openai_server.rerank(request, body)
    assert [entry.id for entry in response.scores] == [
        *(str(i) for i in range(19, 11, -1)), *(str(i) for i in range(12)),
    ]
    assert response.scores[0].score > response.scores[-1].score


async def test_openai_rejects_an_unusable_rank(
    monkeypatch: pytest.MonkeyPatch, rerank_input: tuple[SimpleNamespace, RerankRequest],
) -> None:
    async def post(*_args: object) -> dict:
        return {"status": "completed", "output": [{
            "type": "message", "content": [{"type": "output_text", "text": '{"order":["C019"]}'}],
        }]}

    monkeypatch.setattr(openai_server, "provider_post", post)
    request, body = rerank_input
    with pytest.raises(HTTPException) as exc:
        await openai_server.rerank(request, body)
    assert exc.value.status_code == 502
