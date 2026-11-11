"""Synthetic fixture generators: valid multi-page text PDFs (hand-assembled,
Helvetica/WinAnsi — Latin text), encrypted/corrupt variants, minimal DOCX.
Recreatable test fixtures; nothing is committed as a binary blob.
"""

from __future__ import annotations

import io
import zipfile


def make_text_pdf(page_texts: list[str]) -> bytes:
    """Valid N-page PDF with extractable Latin text (one text block per page)."""
    objs: dict[int, bytes] = {}
    count = len(page_texts)
    page_ids = [4 + 2 * i for i in range(count)]
    content_ids = [5 + 2 * i for i in range(count)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()
    objs[3] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    for i, text in enumerate(page_texts):
        parts = ["BT /F1 11 Tf 50 780 Td 14 TL"]
        for line in text.split("\n"):
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            parts.append(f"({escaped}) Tj T*")
        parts.append("ET")
        stream = "\n".join(parts).encode("latin-1", errors="replace")
        objs[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ids[i]} 0 R >>"
        ).encode()
        objs[content_ids[i]] = (
            f"<< /Length {len(stream)} >>\n".encode() + b"stream\n" + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    total = max(objs) + 1
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objs[oid] + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {total}\n".encode() + b"0000000000 65535 f \n"
    for oid in range(1, total):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def make_encrypted_pdf(password: str = "secret") -> bytes:
    from pypdf import PdfReader, PdfWriter

    source = PdfReader(io.BytesIO(make_text_pdf(["Encrypted page one."])))
    writer = PdfWriter()
    writer.append(source)
    writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_corrupt_pdf() -> bytes:
    # Passes the upload magic-signature check, fails real parsing.
    return b"%PDF-1.4\nthis is not a real pdf structure at all\n"


_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def make_minimal_docx(paragraphs: list[str]) -> bytes:
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()
