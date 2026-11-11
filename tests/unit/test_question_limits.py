"""QuestionRequest contract: the configurable MAX_QUESTION_CHARS limit is enforced
by the route with the standard error envelope; the default (4000) is unchanged."""

from typing import Any

import httpx
import pytest
from app.config.settings import Settings
from app.main import create_app
from app.schemas.qa import QUESTION_HARD_MAX_CHARS, QuestionRequest
from fastapi import FastAPI
from pydantic import ValidationError

URL = "/api/v1/documents/doc-1/questions"


class NoDocsRepo:
    async def get_job_id_for_doc(self, doc_id: str) -> str | None:
        return None


async def _post(app: FastAPI, body: dict[str, Any]) -> httpx.Response:
    app.state.redis_repo = NoDocsRepo()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(URL, json=body)


def _app(**overrides: Any) -> FastAPI:
    return create_app(settings=Settings(_env_file=None, **overrides))  # type: ignore[call-arg]


def test_request_round_trip_and_defaults() -> None:
    request = QuestionRequest.model_validate({"question": "  Какой срок?  "})
    assert request.model_dump() == {"question": "Какой срок?", "debug": False}
    assert QuestionRequest.model_validate(request.model_dump()) == request


async def test_default_limit_is_still_4000() -> None:
    # a request valid under the previous schema (max_length=4000) stays valid:
    # it passes validation and reaches the document lookup (unknown doc -> 404)
    ok = await _post(_app(), {"question": "я" * 4000})
    assert ok.status_code == 404
    too_long = await _post(_app(), {"question": "я" * 4001})
    assert too_long.status_code == 422
    error = too_long.json()["error"]
    assert error["code"] == "VALIDATION_ERROR" and "4000" in error["message"]


async def test_limit_follows_max_question_chars() -> None:
    assert (await _post(_app(max_question_chars=10), {"question": "я" * 11})).status_code == 422
    assert (await _post(_app(max_question_chars=10), {"question": "я" * 10})).status_code == 404
    raised = await _post(_app(max_question_chars=6000), {"question": "я" * 5000})
    assert raised.status_code == 404  # accepted once the operator raises the limit


async def test_contract_mismatch_rejected() -> None:
    # memory fields and blank questions are still rejected
    assert (await _post(_app(), {"question": "q", "history": ["x"]})).status_code == 422
    assert (await _post(_app(), {"question": "   "})).status_code == 422
    with pytest.raises(ValidationError):
        QuestionRequest.model_validate({"question": "я" * (QUESTION_HARD_MAX_CHARS + 1)})


def test_setting_cannot_exceed_schema_cap() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_question_chars=QUESTION_HARD_MAX_CHARS + 1)  # type: ignore[call-arg]
