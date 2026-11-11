"""Server-controlled document file store.

Streams uploads to a temp file with a hard size limit, computes SHA-256 on the
fly, validates content type + magic signature, then atomically renames into
<root>/<doc_id>/original.<ext>. The original filename is metadata only and is
never used as a path. Orphans (files without a Redis doc reference) are removed
by a janitor after a grace period.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

CHUNK_SIZE = 1024 * 1024
TMP_DIR_NAME = ".incoming"

# content type -> (server extension, magic prefix)
ALLOWED_TYPES: dict[str, tuple[str, bytes]] = {
    "application/pdf": ("pdf", b"%PDF-"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "docx",
        b"PK\x03\x04",
    ),
}


class FileTooLargeError(Exception):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds the limit of {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


class UnsupportedFileTypeError(Exception):
    pass


class _Reader(Protocol):
    def __call__(self, size: int) -> Awaitable[bytes]: ...


@dataclass(frozen=True)
class StoredFile:
    doc_id: str
    path: Path
    sha256: str
    size_bytes: int
    content_type: str
    extension: str


class FileStore:
    def __init__(self, root: Path, *, max_bytes: int) -> None:
        self._root = root
        self._max_bytes = max_bytes
        self._tmp_dir = root / TMP_DIR_NAME
        self._root.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def doc_dir(self, doc_id: str) -> Path:
        return self._root / doc_id

    async def save(self, *, doc_id: str, content_type: str, read: _Reader) -> StoredFile:
        if content_type not in ALLOWED_TYPES:
            raise UnsupportedFileTypeError(
                f"unsupported content type '{content_type}'; expected one of: "
                + ", ".join(sorted(ALLOWED_TYPES))
            )
        extension, magic = ALLOWED_TYPES[content_type]
        tmp_path = self._tmp_dir / f"{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        first = b""
        try:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = await read(CHUNK_SIZE)
                    if not chunk:
                        break
                    if not first:
                        first = chunk[:16]
                    size += len(chunk)
                    if size > self._max_bytes:
                        raise FileTooLargeError(self._max_bytes)
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0 or not first.startswith(magic):
                raise UnsupportedFileTypeError(
                    f"file signature does not match declared type '{content_type}'"
                )
            final_dir = self.doc_dir(doc_id)
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"original.{extension}"
            os.replace(tmp_path, final_path)  # atomic within the same volume
            return StoredFile(
                doc_id=doc_id,
                path=final_path,
                sha256=digest.hexdigest(),
                size_bytes=size,
                content_type=content_type,
                extension=extension,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def delete_doc_files(self, doc_id: str) -> None:
        shutil.rmtree(self.doc_dir(doc_id), ignore_errors=True)

    def cleanup_orphans(
        self, *, is_referenced: Callable[[str], bool], grace_s: float = 60.0
    ) -> list[str]:
        """Remove doc dirs with no Redis reference and stale temp files. Returns removed ids."""
        removed: list[str] = []
        now = time.time()
        for entry in self._root.iterdir():
            if entry.name == TMP_DIR_NAME or not entry.is_dir():
                continue
            if now - entry.stat().st_mtime < grace_s:
                continue
            if not is_referenced(entry.name):
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
        for stale in self._tmp_dir.glob("*.part"):
            if now - stale.stat().st_mtime >= grace_s:
                stale.unlink(missing_ok=True)
        return removed
