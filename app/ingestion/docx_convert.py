"""DOCX -> PDF via an isolated LibreOffice process.

Unique user profile per job, hard timeout, fixed fonts baked into the image
(plan §5.1). Citation pages always refer to the resulting server-side PDF.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
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
    with tempfile.TemporaryDirectory(prefix="lo-profile-") as profile_dir:
        out_dir = Path(tempfile.mkdtemp(prefix="lo-out-"))
        try:
            command = [
                "soffice",
                "--headless",
                "--norestore",
                f"-env:UserInstallation=file://{profile_dir}/{uuid.uuid4().hex}",
                "--convert-to",
                "pdf:writer_pdf_Export",
                "--outdir",
                str(out_dir),
                str(docx_path),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, timeout=CONVERT_TIMEOUT_S, check=False
                )
            except subprocess.TimeoutExpired as exc:
                raise ConversionTimeoutError(
                    f"LibreOffice conversion exceeded {CONVERT_TIMEOUT_S}s"
                ) from exc
            produced = out_dir / (docx_path.stem + ".pdf")
            if completed.returncode != 0 or not produced.exists():
                stderr = completed.stderr.decode("utf-8", errors="replace")[-500:]
                raise ConversionFailedError(
                    "LibreOffice failed to convert DOCX to PDF",
                    detail=stderr or f"exit={completed.returncode}",
                )
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), output_pdf)
            return output_pdf
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
