"""Internal LLM contracts (§7.2): atomic claims with evidence, and the verifier
verdict. The LLM NEVER produces the external object with pages/sections — the
server derives citations from authoritative chunk payload.
"""

from __future__ import annotations

import enum
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Short server-issued source id ([source-N]); mapped back to the real chunk.
    chunk_id: str
    quote: str = Field(min_length=1)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)  # every claim needs evidence

    @field_validator("text")
    @classmethod
    def _server_citation_markers(cls, value: str) -> str:
        if re.search(r"\[\d+\]", value):
            raise ValueError("citation markers belong in evidence, not claim text")
        return value


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool
    claims: list[Claim]
    missing_information: list[str]

    @model_validator(mode="after")
    def _consistent_answer(self) -> DraftAnswer:
        if self.answerable != bool(self.claims) or (self.answerable and self.missing_information):
            raise ValueError("answerable requires claims covering all requested information")
        return self


DRAFT_JSON_SCHEMA: dict = {
    "name": "draft_answer",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["answerable", "claims", "missing_information"],
        "properties": {
            "answerable": {"type": "boolean"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "evidence"],
                    "properties": {
                        "text": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["chunk_id", "quote"],
                                "properties": {
                                    "chunk_id": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "missing_information": {"type": "array", "items": {"type": "string"}},
        },
    },
}


class ClaimVerdict(enum.StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"


class Coverage(enum.StrEnum):
    FULL = "full"
    PARTIAL = "partial"


class VerdictEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_index: int = Field(ge=0)
    verdict: ClaimVerdict


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[VerdictEntry]
    coverage: Coverage


VERIFY_JSON_SCHEMA: dict = {
    "name": "verification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts", "coverage"],
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_index", "verdict"],
                    "properties": {
                        "claim_index": {"type": "integer", "minimum": 0},
                        "verdict": {
                            "type": "string",
                            "enum": ["supported", "unsupported", "insufficient"],
                        },
                    },
                },
            },
            "coverage": {"type": "string", "enum": ["full", "partial"]},
        },
    },
}
