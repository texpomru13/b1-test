"""External QA contracts (§9.1-9.2): request and answer with server-enforced
invariants. Citations are built by the SERVER from chunk payload and recovered
quote offsets — these invariants are never delegated to the LLM.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

NO_ANSWER_MESSAGE = (
    "В документе не найдено достаточно сведений для ответа на этот вопрос."
)

_MARKER = re.compile(r"\[(\d+)\]")

# Mirrors the upper bound of Settings.max_question_chars.
QUESTION_HARD_MAX_CHARS = 100_000


class QuestionRequest(BaseModel):
    # extra='forbid': history/thread_id/messages and any memory fields are rejected.
    model_config = ConfigDict(extra="forbid")

    # Structural cap only; the configurable limit MAX_QUESTION_CHARS is enforced
    # by the route (422 VALIDATION_ERROR) because it comes from Settings.
    question: str = Field(min_length=1, max_length=QUESTION_HARD_MAX_CHARS)
    debug: bool = False

    @model_validator(mode="after")
    def _trim(self) -> QuestionRequest:
        trimmed = self.question.strip()
        if not trimmed:
            raise ValueError("question must not be blank")
        object.__setattr__(self, "question", trimmed)
        return self


class Citation(BaseModel):
    id: int
    text: str  # exact quote from the canonical document text
    page: int = Field(gt=0)  # 1-based physical page of the quote start
    page_to: int = Field(gt=0)
    section: str | None
    chunk_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def _span(self) -> Citation:
        if self.char_end <= self.char_start:
            raise ValueError("citation char_end must be greater than char_start")
        if self.page_to < self.page:
            raise ValueError("citation page_to must be >= page")
        return self


class QaDiagnostics(BaseModel):
    dense_returned: int | None
    sparse_returned: int | None
    fused_returned: int
    reranked_count: int
    context_count: int
    context_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    rerank_scores: dict[str, float]
    stage_ms: dict[str, int]
    model_ids: dict[str, str]
    token_usage: dict[str, int]
    branch_counts_source: str | None
    diagnostic_warnings: list[str]


class AnswerResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    diagnostics: QaDiagnostics | None = None  # present ONLY when debug=true

    @model_validator(mode="after")
    def _invariants(self) -> AnswerResponse:
        if self.found:
            if not self.citations:
                raise ValueError("found=true requires non-empty citations")
            if not self.answer.strip():
                raise ValueError("found=true requires a non-empty answer")
            ids = [c.id for c in self.citations]
            if len(ids) != len(set(ids)):
                raise ValueError("citation ids must be unique")
            markers = {int(m) for m in _MARKER.findall(self.answer)}
            known = set(ids)
            if not markers:
                raise ValueError("found=true answer must reference citations via [n]")
            if not markers.issubset(known):
                raise ValueError(f"answer references unknown citations: {markers - known}")
        else:
            if self.citations:
                raise ValueError("found=false requires empty citations")
        return self
