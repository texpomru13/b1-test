import hashlib
from pathlib import Path

import pytest
from app.storage.file_store import (
    FileStore,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def reader_for(data: bytes):
    view = memoryview(data)
    state = {"pos": 0}

    async def read(size: int) -> bytes:
        start = state["pos"]
        chunk = bytes(view[start : start + size])
        state["pos"] = start + len(chunk)
        return chunk

    return read


async def test_save_streams_hashes_and_renames(tmp_path: Path) -> None:
    store = FileStore(tmp_path, max_bytes=1024)
    stored = await store.save(
        doc_id="doc1", content_type="application/pdf", read=reader_for(PDF_BYTES)
    )
    assert stored.path == tmp_path / "doc1" / "original.pdf"
    assert stored.path.read_bytes() == PDF_BYTES
    assert stored.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert stored.size_bytes == len(PDF_BYTES)
    assert list((tmp_path / ".incoming").glob("*")) == []  # no temp leftovers


async def test_size_limit_enforced_and_nothing_left(tmp_path: Path) -> None:
    store = FileStore(tmp_path, max_bytes=10)
    with pytest.raises(FileTooLargeError):
        await store.save(
            doc_id="doc1", content_type="application/pdf", read=reader_for(PDF_BYTES)
        )
    assert not (tmp_path / "doc1").exists()
    assert list((tmp_path / ".incoming").glob("*")) == []


async def test_signature_mismatch_rejected(tmp_path: Path) -> None:
    store = FileStore(tmp_path, max_bytes=1024)
    with pytest.raises(UnsupportedFileTypeError):
        await store.save(
            doc_id="doc1",
            content_type="application/pdf",
            read=reader_for(b"PK\x03\x04 this is actually a zip"),
        )
    assert not (tmp_path / "doc1").exists()


async def test_unknown_content_type_rejected(tmp_path: Path) -> None:
    store = FileStore(tmp_path, max_bytes=1024)
    with pytest.raises(UnsupportedFileTypeError):
        await store.save(doc_id="d", content_type="text/plain", read=reader_for(b"hello"))


async def test_janitor_removes_only_unreferenced_after_grace(tmp_path: Path) -> None:
    store = FileStore(tmp_path, max_bytes=1024)
    await store.save(doc_id="orphan", content_type="application/pdf", read=reader_for(PDF_BYTES))
    await store.save(doc_id="kept", content_type="application/pdf", read=reader_for(PDF_BYTES))

    removed = store.cleanup_orphans(is_referenced=lambda doc_id: doc_id == "kept", grace_s=0)
    assert removed == ["orphan"] or set(removed) == {"orphan"}
    assert not (tmp_path / "orphan").exists()
    assert (tmp_path / "kept" / "original.pdf").exists()

    # Fresh files inside the grace period are never touched.
    await store.save(doc_id="fresh", content_type="application/pdf", read=reader_for(PDF_BYTES))
    assert store.cleanup_orphans(is_referenced=lambda _: False, grace_s=3600) == []
    assert (tmp_path / "fresh").exists()
