"""Combine the document question banks and render their public catalog."""
from __future__ import annotations

import argparse
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path

from paths import ROOT, WORK
from qualification_data import digest, write_once

DOCUMENTS = ROOT / "test_documents"
GOLD = ROOT / "eval/gold"
PACKS = ROOT / "eval/packs"
PACK_NAMES = ("contracts-six", "published-six", "external-six")
# Display names for files whose stable fixture IDs are publisher-neutral.
TITLES = {
    "bulletin": "Вестник Банка России № 25",
    "echr": "Европейская конвенция о правах человека",
    "marrakesh": "Марракешский договор",
    "nairobi": "Найробский договор об охране олимпийского символа",
    "nice": "Ниццкое соглашение",
    "space": "Договоры и принципы ООН по космосу",
    "alferov": "Договор на обучение в университете Алфёрова",
    "hotel": "Оферта гостиницы в Сочи",
    "moreon": "Абонемент комплекса «Мореон»",
    "roerich": "Тарифы музея семьи Рерихов",
    "troika": "Оферта транспортной карты «Тройка»",
    "warehouse": "Договор аренды ячейки «ГлавСклад»",
}


def documents() -> dict[str, Path]:
    files = [p for p in DOCUMENTS.iterdir() if p.is_file() and p.suffix in {".pdf", ".docx"}]
    result = {p.stem.partition("_")[0]: p.with_name(unicodedata.normalize("NFC", p.name))
              for p in files}
    if len(result) != len(files):
        raise ValueError("duplicate numeric document identifier")
    for key in PACK_NAMES:
        for fid, ref in json.loads((PACKS / key / "documents.json").read_text()).items():
            path = (ROOT / ref).resolve()
            if fid in result and not result[fid].samefile(path):
                raise ValueError(f"fixture identifier belongs to two files: {fid}")
            result[fid] = path
    if len(set(result.values())) != len(result):
        raise ValueError("same document has multiple fixture identifiers")
    return result


def banks() -> list[Path]:
    return [GOLD / "gold.jsonl", GOLD / "scan-06.jsonl", *(
        PACKS / key / "gold.jsonl" for key in PACK_NAMES)]


def cases(docs: dict[str, Path]) -> list[dict]:
    rows = [json.loads(line) for source in banks() for line in source.read_text().splitlines()]
    ids = [row["case_id"] for row in rows]
    questions = {(row["doc_fixture_id"], " ".join(row["question"].casefold().split()))
                 for row in rows}
    if len(set(ids)) != len(rows) or len(questions) != len(rows):
        raise ValueError("duplicate case ID or question for the same document")
    if {row["doc_fixture_id"] for row in rows} != set(docs):
        raise ValueError("the catalog and the questions cover different documents")
    return rows


def lock(docs: dict[str, Path]) -> dict:
    return {"documents": {fid: digest(path) for fid, path in sorted(docs.items())},
            "questions": {p.relative_to(ROOT).as_posix(): digest(p) for p in banks()}}


def prepare_gold() -> dict[str, Path]:
    docs = documents()
    rows = cases(docs)
    if lock(docs) != json.loads((GOLD / "catalog-lock.json").read_text()):
        raise ValueError("a document or question bank differs from the catalog lock")
    if WORK == ROOT / "eval/work":
        raise ValueError("select a dedicated B1_EVAL_WORK_DIR for the combined catalog")
    WORK.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    write_once(WORK / "gold.jsonl", data.encode())
    return docs


def check_index(docs: dict[str, Path], rows: list[dict]) -> None:
    corpus = json.loads((WORK / "corpus.json").read_text())["docs"]
    for fid, path in docs.items():
        index = json.loads((WORK / "index" / f"{fid}.json").read_text())
        if (corpus[fid]["sha256"] != digest(path) or index["doc_id"] != corpus[fid]["doc_id"]
                or not index["accepted_run_id"]):
            raise ValueError(f"unexpected file or unpublished index: {fid}")
        for row in (r for r in rows if r["doc_fixture_id"] == fid):
            for span in row.get("evidence", row["gold_spans"]):
                for o in span["occurrences"]:
                    page = index["pages"][o["page"] - 1]
                    a, b = o["char_start"], o["char_end"]
                    if (not page["char_start"] <= a < page["char_end"]
                            or index["canonical_text"][a:b] != span["text"]):
                        raise ValueError(f"page or evidence differs in {row['case_id']}")


def render(docs: dict[str, Path], rows: list[dict]) -> None:
    counts = Counter(r["doc_fixture_id"] for r in rows)
    names = {fid: TITLES.get(fid) or path.stem.partition("_")[2].replace("_", " ")
             for fid, path in docs.items()}
    overview = ["# Документы и вопросы", "",
                f"Реальных документов — **{len(docs)}**, вопросов — **{len(rows)}**. "
                "Каждая строка ведёт к файлу и его вопросам с эталонными ответами. "
                "[Единый список вопросов](../eval/gold/questions.md)", "",
                "| Документ | Формат | Вопросы и эталоны |", "|---|---|---:|"]
    sheet = ["# Вопросы и эталонные ответы", "",
             "Страницы PDF нумеруются с единицы. Для DOCX указана физическая страница "
             "после преобразования в PDF с помощью LibreOffice. В API эта копия доступна "
             "по маршруту документа `/normalized`. "
             "[Каталог файлов](../../test_documents/README.md)", ""]
    for fid in sorted(docs, key=lambda k: names[k].casefold()):
        path = docs[fid]
        ref = os.path.relpath(path, DOCUMENTS)
        overview.append(f"| [{names[fid]}](<{ref}>) | {path.suffix[1:].upper()} | "
                        f"[{counts[fid]}](../eval/gold/questions.md#doc-{fid}) |")
        src = os.path.relpath(path, GOLD)
        sheet += [f'<a id="doc-{fid}"></a>', "", f"## [{names[fid]}](<{src}>)", "",
                  "| ID | Вопрос | Ожидаемый ответ | Страницы |", "|---|---|---|---|"]
        for row in (r for r in rows if r["doc_fixture_id"] == fid):
            question = row["question"]
            if row["answerable"]:
                answer = row["reference_answer"].strip().replace(".. ", ". ")
                pages = sorted({o["page"] for s in row["gold_spans"] for o in s["occurrences"]})
                refs = ", ".join(f"[{p}](<{src}#page={p}>)" if path.suffix == ".pdf" else str(p)
                                 for p in pages)
            else:
                answer, refs = "Достаточных сведений в документе нет.", "—"
            clean = [" ".join(x.split()).replace("|", r"\|") for x in (question, answer, refs)]
            sheet.append(f"| {row['case_id']} | " + " | ".join(clean) + " |")
        sheet.append("")
    overview += ["", "Подробные листы сверки с дословными выдержками охватывают "
                 "[договоры организаций](../eval/packs/contracts-six/source-review.md), "
                 "[официальные публикации](../eval/packs/published-six/source-review.md) и "
                 "[публичные оферты и тарифы](../eval/packs/external-six/source-review.md).", "",
                 "Для контролируемых тестов правил и таблиц предусмотрены "
                 "[синтетические PDF](synthetic-six) и "
                 "[60 вопросов к ним](../eval/qualification/source-review.md).", "",
                 "Независимая выборка содержит [10 новых PDF](independent-ten) "
                 "и [80 вопросов](../eval/packs/independent-ten/source-review.md). "
                 "Она оценивается своим прогоном.", ""]
    (DOCUMENTS / "README.md").write_text("\n".join(overview))
    (GOLD / "questions.md").write_text("\n".join(sheet))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true",
                        help="update Markdown from the locked questions")
    parser.add_argument("--check-index", action="store_true", help="verify the exported live index")
    args = parser.parse_args()
    docs = documents()
    rows = cases(docs)
    if lock(docs) != json.loads((GOLD / "catalog-lock.json").read_text()):
        raise ValueError("catalog inputs differ from the lock")
    if args.render:
        render(docs, rows)
    if args.check_index:
        prepare_gold()
        check_index(docs, rows)
    positive = sum(r["answerable"] for r in rows)
    print(f"{len(docs)} documents, {len(rows)} questions, {positive} answerable")


if __name__ == "__main__":
    main()
