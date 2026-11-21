"""Freeze a new annotated corpus once. Later invocations verify the same inputs."""
from __future__ import annotations

import argparse
import json
import os

from qualification_data import (
    PACK_SIZES,
    build_canonical,
    digest,
    extract_pdf,
    locked_files,
    pack_path,
    qualification_cases,
    source_files,
    source_sheet,
    write_once,
)


def compile_case(raw: dict, source) -> dict:
    spans: list[dict] = []

    def materialize(items: list[dict]) -> list[dict]:
        converted = []
        for item in items:
            refs = []
            for quote in item["quotes"]:
                if not quote.strip():
                    raise ValueError(f"{raw['case_id']}: empty quote")
                starts = []
                cursor = 0
                while (pos := source.text.find(quote, cursor)) >= 0:
                    starts.append(pos)
                    cursor = pos + 1
                if not starts:
                    raise ValueError(f"{raw['case_id']}: missing quote {quote[:80]!r}")
                span = {"text": quote, "occurrences": [
                    {"char_start": p, "char_end": p + len(quote), "page": source.page_of(p)}
                    for p in starts]}
                if span not in spans:
                    spans.append(span)
                refs.append(spans.index(span) + 1)
            if not refs or not item["text"].strip():
                raise ValueError(f"{raw['case_id']}: fact lacks text or evidence")
            converted.append({"text": item["text"], "span_ids": refs})
        return converted

    required = materialize(raw["required_facts"])
    conditions = materialize(raw["required_conditions"])
    required_spans = list(spans)
    extras = materialize(raw["optional_facts"])
    expected = [f["text"] for f in required + conditions]
    if bool(expected) == (raw["category"] == "no_answer"):
        raise ValueError(f"{raw['case_id']}: inconsistent answerability")
    return {
        **raw, "required_facts": required, "required_conditions": conditions,
        "optional_facts": extras, "expected_facts": expected, "answerable": bool(expected),
        "evidence": spans, "gold_spans": required_spans, "reference_answer": ". ".join(expected),
        "split": "holdout", "provenance": "agent-authored; awaiting human source review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[n for n in PACK_SIZES if n != "qualification"])
    args = parser.parse_args()
    pack = pack_path(args.name)
    docs = source_files(pack)
    sources = {fid: build_canonical(extract_pdf(p, max_pages=100).pages) for fid, p in docs.items()}
    cases = []
    for line in (pack / "drafts.jsonl").read_text().splitlines():
        draft = json.loads(line)
        cases.append(compile_case(draft, sources[draft["doc_fixture_id"]]))
    count, per_file = PACK_SIZES[pack.name]
    total = count * per_file
    if (len(cases) != total or len({c["case_id"] for c in cases}) != total
            or len({(c["doc_fixture_id"], c["question"]) for c in cases}) != total
            or sum(not c["answerable"] for c in cases) < 20
            or any(sum(c["doc_fixture_id"] == fid for c in cases) != per_file for fid in docs)):
        raise ValueError(f"expected {per_file} unique questions per document, at least 20 absences")
    gold = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    write_once(pack / "gold.jsonl", gold.encode())
    hashes = {os.path.relpath(p, pack): digest(p) for p in locked_files(pack, docs)}
    lock = json.dumps({"version": 1, "files": hashes}, indent=2) + "\n"
    write_once(pack / "lock.json", lock.encode())
    qualification_cases(pack)
    sheet = pack / "source-review.md"
    if not sheet.exists():
        write_once(sheet, source_sheet(cases, sheet, pack).encode())
    print(f"verified {pack.name}: {len(cases)} questions, {len(docs)} documents")


if __name__ == "__main__":
    main()
