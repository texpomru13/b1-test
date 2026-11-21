"""Read a single physical page of a scanned PDF with local Poppler and Tesseract."""
from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
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
