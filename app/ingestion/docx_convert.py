"""DOCX -> PDF via an isolated LibreOffice process.

Unique user profile per job, hard timeout, fixed fonts baked into the image
(plan §5.1). Citation pages always refer to the resulting server-side PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from app.ingestion.errors import ConversionFailedError, ConversionTimeoutError

CONVERT_TIMEOUT_S = 120


def soffice_available() -> bool:
    return shutil.which("soffice") is not None


def convert_docx_to_pdf(docx_path: Path, output_pdf: Path) -> Path:
    if not soffice_available():
        raise ConversionFailedError(
            "LibreOffice (soffice) is not available in this environment"
        )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Produce the PDF on the document volume so a concurrent download sees
    # either the complete previous file or the complete new conversion.
    with tempfile.TemporaryDirectory(prefix=".convert-", dir=output_pdf.parent) as work:
        directory = Path(work)
        command = [
            "soffice", "--headless", "--norestore",
            f"-env:UserInstallation={(directory / 'profile').resolve().as_uri()}",
            "--convert-to", "pdf:writer_pdf_Export",
            "--outdir", str(directory), str(docx_path),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=CONVERT_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ConversionTimeoutError(
                f"LibreOffice conversion exceeded {CONVERT_TIMEOUT_S}s"
            ) from exc
        produced = directory / (docx_path.stem + ".pdf")
        if completed.returncode != 0 or not produced.is_file():
            stderr = completed.stderr.decode("utf-8", errors="replace")[-500:]
            raise ConversionFailedError(
                "LibreOffice failed to convert DOCX to PDF",
                detail=stderr or f"exit={completed.returncode}",
            )
        return produced.replace(output_pdf)
