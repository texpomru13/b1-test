"""Read a single physical page of a scanned PDF with local Poppler and Tesseract."""
from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache, partial
from pathlib import Path

from app.ingestion.errors import OcrRequiredError

LANGUAGES = "rus+eng"


@lru_cache(maxsize=1)
def _check_languages() -> None:
    available = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                               check=True, timeout=10, text=True).stdout.splitlines()
    if set(LANGUAGES.split("+")) - set(available):
        raise OcrRequiredError(f"local Tesseract needs the {LANGUAGES} language packs")


def recognize_page(pdf: Path, page: int) -> str:
    try:
        _check_languages()
        with tempfile.TemporaryDirectory(prefix="b1-ocr-") as directory:
            prefix = Path(directory) / "page"
            # Cap the longest side: PDF media boxes may have arbitrary dimensions.
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                            "-gray", "-scale-to", "3200", str(pdf.resolve()), str(prefix)],
                           check=True, capture_output=True, timeout=30)
            result = subprocess.run(
                ["tesseract", str(prefix.with_suffix(".pgm")), "stdout", "-l", LANGUAGES],
                env={**os.environ, "OMP_THREAD_LIMIT": "1"}, check=True,
                capture_output=True, encoding="utf-8", timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OcrRequiredError(f"OCR failed on physical page {page}: {type(exc).__name__}") from exc
    return result.stdout.strip()


def recognize_page_openai(pdf: Path, page: int, *, key: str) -> str:
    import base64

    import httpx

    from app.inference.openai_api import (
        API_URL,
        OCR_MODEL,
        auth_headers,
        response_text,
        sync_response,
    )
    from app.reliability.policy import retry_sync

    try:
        with tempfile.TemporaryDirectory(prefix="b1-ocr-") as directory:
            prefix = Path(directory) / "page"
            subprocess.run(
                ["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                 "-gray", "-scale-to", "3200", "-png", str(pdf.resolve()), str(prefix)],
                check=True, capture_output=True, timeout=30,
            )
            encoded = base64.b64encode(prefix.with_suffix(".png").read_bytes()).decode("ascii")
        with httpx.Client(base_url=API_URL, headers=auth_headers(key), timeout=120) as client:
            payload = {
                "model": OCR_MODEL, "store": False, "max_output_tokens": 8192,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": (
                        "Transcribe this page verbatim, keeping reading order and line breaks. "
                        "Preserve original spelling, punctuation and numbers. "
                        "Do not translate, summarize, reconstruct illegible text or add Markdown. "
                        "Return empty text if there is no readable text."
                    )},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}",
                     "detail": "high"},
                ]}],
            }
            result = retry_sync(partial(sync_response, client, payload, "ocr"),
                                op_name="openai.ocr", max_attempts=2)
        return response_text(result).strip()
    except (OSError, subprocess.SubprocessError, httpx.HTTPError, ValueError) as exc:
        raise OcrRequiredError(f"OpenAI OCR failed on page {page}: {type(exc).__name__}") from exc
