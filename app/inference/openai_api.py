"""Responses API helpers and per-request usage accounting."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx

API_URL = "https://api.openai.com/v1"
OCR_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gpt-6.1-terra"


def auth_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def ledger_row(response: httpx.Response, operation: str) -> dict[str, Any]:
    result = response.json()
    usage = result.get("usage") or {}
    return {
        "request_id": response.headers.get("x-request-id") or result.get("id"),
        "operation": operation,
        "model": result.get("model"),
        "status": result.get("status", response.status_code),
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", 0),
        "input_details": usage.get("input_tokens_details") or {},
        "output_details": usage.get("output_tokens_details") or {},
        "service_tier": result.get("service_tier"),
    }


def record_usage(response: httpx.Response, operation: str) -> dict[str, Any]:
    path = os.environ.get("OPENAI_USAGE_PATH")
    if path:
        row = ledger_row(response, operation)
        line = json.dumps(row, separators=(",", ":")).encode() + b"\n"
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    return response.json()


async def record_qa_usage(response: httpx.Response) -> None:
    if response.is_success and response.request.url.path.endswith("/responses"):
        await response.aread()
        record_usage(response, "qa")


def response_text(result: dict[str, Any]) -> str:
    if result.get("status") != "completed":
        raise ValueError("OpenAI did not complete the response")
    return "".join(
        part["text"]
        for item in result["output"] if item["type"] == "message"
        for part in item["content"] if part["type"] == "output_text"
    )


def sync_response(
    client: httpx.Client, payload: dict[str, Any], operation: str,
) -> dict[str, Any]:
    response = client.post("responses", json=payload)
    response.raise_for_status()
    return record_usage(response, operation)


class OpenAIJudge:
    def __init__(self, model: str, request: Callable[[dict], dict]) -> None:
        self.model = model
        self._request = request

    def __call__(self, prompt: str) -> str:
        return response_text(self._request({
            "model": self.model,
            "input": prompt,
            "store": False,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 4096,
            "text": {"format": {"type": "json_object"}},
        }))
