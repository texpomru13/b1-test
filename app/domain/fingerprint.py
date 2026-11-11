"""Upload request fingerprint: file content + title + processing settings.

Same idempotency key with the same fingerprint returns prior IDs; a different
fingerprint under the same key is a 409 conflict (IMPLEMENTATION_PLAN.md §8.2).
"""

from __future__ import annotations

import hashlib

from app.config.settings import Settings


def upload_fingerprint(*, file_sha256: str, title: str | None, settings: Settings) -> str:
    processing = "|".join(
        [
            f"chunk={settings.chunk_tokens}",
            f"overlap={settings.chunk_overlap_tokens}",
            f"lexnorm=v{settings.lexical_normalizer_version}",
            f"instr=v{settings.embedding_instruction_version}",
        ]
    )
    payload = f"{file_sha256}|{title or ''}|{processing}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
