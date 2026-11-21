"""Verify the locked qualification corpus against the exact text visible to the indexer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from paths import ROOT, WORK

sys.path.insert(0, str(ROOT))
from app.ingestion.canonical import build_canonical
from app.ingestion.pdf_loader import extract_pdf


def pack_path(name: str) -> Path:
    if name == "qualification":
        return ROOT / "eval" / "qualification"
    if name in {"original-six", "published-six", "external-six"}:
        return ROOT / "eval" / "packs" / name
    raise ValueError(f"unknown B1_EVAL_PACK: {name}")


PACK = pack_path(os.environ.get("B1_EVAL_PACK", "qualification"))


def source_files(pack: Path = PACK) -> dict[str, Path]:
    manifest = pack / "documents.json"
    if manifest.exists():
        refs = json.loads(manifest.read_text())
        docs = {k: (ROOT / v).resolve() for k, v in refs.items()}
    else:
        docs = {p.stem: p for p in sorted((pack / "fixtures").glob("*.pdf"))}
    if len(docs) != 6 or len(set(docs.values())) != 6 or any(
        not p.is_relative_to(ROOT) or p.suffix.lower() != ".pdf" for p in docs.values()
    ):
        raise ValueError("expected six distinct PDFs in the repository")
    return docs


def locked_files(pack: Path, docs: dict[str, Path]) -> list[Path]:
    inputs = [pack / "gold.jsonl", *docs.values()]
    if (pack / "documents.json").exists():
        inputs.append(pack / "documents.json")
    return inputs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualification_cases(pack: Path = PACK) -> list[dict]:
    files = json.loads((pack / "lock.json").read_text())["files"]
    docs = source_files(pack)
    if set(files) != {os.path.relpath(p, pack) for p in locked_files(pack, docs)}:
        raise ValueError("qualification corpus files differ from the lock")
    for name, expected in files.items():
        if digest(pack / name) != expected:
            raise ValueError(f"qualification input changed: {name}")

    canonical = {fid: build_canonical(extract_pdf(p, max_pages=100).pages)
                 for fid, p in docs.items()}
    cases = [json.loads(line) for line in (pack / "gold.jsonl").read_text().splitlines()]
    if (len({c["case_id"] for c in cases}) != len(cases) or len(cases) != 60
            or len({(c["doc_fixture_id"], c["question"]) for c in cases}) != len(cases)
            or sum(not c["answerable"] for c in cases) < 20
            or set(canonical) != {c["doc_fixture_id"] for c in cases}):
        raise ValueError("incomplete or duplicate qualification cases")
    for c in cases:
        source = canonical[c["doc_fixture_id"]]
        for span in c["evidence"]:
            if not span["occurrences"]:
                raise ValueError(f"missing occurrence in {c['case_id']}")
            for occurrence in span["occurrences"]:
                start, end = occurrence["char_start"], occurrence["char_end"]
                if (source.text[start:end] != span["text"]
                        or source.page_of(start) != occurrence["page"]):
                    raise ValueError(f"source/evidence mismatch in {c['case_id']}")
        required = c["required_facts"] + c["required_conditions"]
        if bool(required) != c["answerable"] or not c["scope"]:
            raise ValueError(f"incomplete rubric in {c['case_id']}")
        for fact in required + c["optional_facts"]:
            if (not fact["span_ids"] or not all(1 <= i <= len(c["evidence"])
                                                for i in fact["span_ids"])):
                raise ValueError(f"missing fact evidence in {c['case_id']}")
    return cases


def write_once(path: Path, data: bytes) -> None:
    """Publish a whole input atomically. Concurrent different inputs cannot replace it."""
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"existing file contains different data: {path}") from None


def check_export(cases: list[dict]) -> None:
    """A different OCR/parser build may move spans even for the exact same PDF."""
    if digest(WORK / "gold.jsonl") != digest(PACK / "gold.jsonl"):
        raise ValueError("export and work directory belong to different packs")
    corpus = json.loads((WORK / "corpus.json").read_text())["docs"]
    loaded = {fid: json.loads((WORK / "index" / f"{fid}.json").read_text())
              for fid in source_files()}
    for fid, index in loaded.items():
        if index["doc_id"] != corpus[fid]["doc_id"] or not index["accepted_run_id"]:
            raise ValueError(f"stale document or unpublished index: {fid}")
    for c in cases:
        index = loaded[c["doc_fixture_id"]]
        for span in c["evidence"]:
            for o in span["occurrences"]:
                page = index["pages"][o["page"] - 1]
                if not (page["char_start"] <= o["char_start"] < page["char_end"]
                        and index["canonical_text"][o["char_start"]:o["char_end"]] == span["text"]):
                    raise ValueError(f"{c['case_id']}: exported text/page disagrees with gold")


def install_gold() -> None:
    qualification_cases()
    dst = WORK / "gold.jsonl"
    if WORK == ROOT / "eval" / "work" or (dst.exists() and digest(dst) != digest(PACK / dst.name)):
        raise ValueError("select a separate B1_EVAL_WORK_DIR for qualification")
    if (WORK / "corpus.json").exists():
        corpus = json.loads((WORK / "corpus.json").read_text())
        if set(corpus["docs"]) - source_files().keys():
            raise ValueError("existing work directory contains another corpus")
    WORK.mkdir(parents=True, exist_ok=True)
    write_once(dst, (PACK / dst.name).read_bytes())


def source_sheet(cases: list[dict], output: Path, pack: Path = PACK) -> str:
    docs = source_files(pack)
    lines = [
        "# Лист независимой сверки",
        "",
        "Проверяющий (имя):",
        "",
        f"SHA-256 фиксации корпуса: `{digest(pack / 'lock.json')}`",
        "",
        "Откройте PDF каждого документа. Для положительного вопроса сверьте объект, значения, "
        "исключения и связи между строками таблицы. Проверьте, что каждый обязательный факт "
        "следует из указанных доказательств и нужен для ответа. Для отрицательного вопроса "
        "прочитайте весь документ и убедитесь, что ответ отсутствует.",
        "",
        "Укажите имя выше. Отметьте `[x]` у каждого согласованного вопроса. При расхождении "
        "оставьте флажок пустым и запишите замечание под ним. Заполненную копию можно передать в "
        "`qualification_review.py score --source-review ПУТЬ`.",
    ]
    last_doc = ""
    for case in cases:
        fid = case["doc_fixture_id"]
        if fid != last_doc:
            path = Path(os.path.relpath(docs[fid], output.parent))
            lines += ["", f"## Документ [{docs[fid].name}](<{path.as_posix()}>)", ""]
            last_doc = fid
        lines += ["", f"### {case['case_id']}", "", case["question"], "",
                  f"Проверяемый объект — {case['scope']}.", ""]
        if case["answerable"]:
            for field, title in (("required_facts", "Нужные для ответа сведения"),
                                 ("required_conditions", "Условия применения"),
                                 ("optional_facts", "Дополнительные сведения")):
                if not case[field]:
                    continue
                lines += [title, ""]
                for item in case[field]:
                    refs = ", ".join(str(i) for i in item["span_ids"])
                    lines.append(f"- {item['text']} (выдержки {refs}).")
                lines.append("")
            lines.append("Доказательства в PDF")
            lines.append("")
            for i, span in enumerate(case["evidence"], 1):
                pages = ", ".join(str(o["page"]) for o in span["occurrences"])
                quote = " ".join(span["text"].splitlines())
                lines.append(f"{i}. Стр. {pages}. «{quote}»")
            if case["forbidden_facts"]:
                lines += ["", "Одна из проверяемых ошибок — " + ", ".join(case["forbidden_facts"])]
        else:
            lines.append("Ожидается отказ. Для проверки отсутствия ответа откройте весь документ.")
        lines += ["", f"- [ ] {case['case_id']} сверено", "", "Замечание:", ""]
    return "\n".join(lines)


def source_reviewer(path: Path, case_ids: set[str]) -> str | None:
    data = path.read_text(encoding="utf-8")
    if f"SHA-256 фиксации корпуса: `{digest(PACK / 'lock.json')}`" not in data:
        raise ValueError("source review belongs to a different qualification lock")
    checked = re.findall(r"^- \[[xX]\] ([\w-]+) сверено$", data, re.M)
    reviewer = re.search(r"^Проверяющий \(имя\):[ \t]*(.*)$", data, re.M)
    if len(checked) == len(case_ids) and set(checked) == case_ids and reviewer:
        return reviewer[1].strip() or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sheet", type=Path, help="write a Markdown source review sheet")
    parser.add_argument("--check-index", action="store_true", help="check exported coordinates")
    args = parser.parse_args()
    cases = qualification_cases()
    if args.check_index:
        check_export(cases)
    if args.source_sheet:
        write_once(args.source_sheet, source_sheet(cases, args.source_sheet).encode())
    print(f"{len(cases)} questions, {sum(c['answerable'] for c in cases)} answerable; "
          "all locked source spans and pages verified")


if __name__ == "__main__":
    main()
