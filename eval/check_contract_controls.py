"""Check the answer text and cited source coordinates of four contract acceptance cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from paths import WORK

CHECKS = {
    "14-03": [
        ("Стоимость и НДС", r"(?=.*\b6\s?600\b)(?=.*\bс НДС)"),
        ("Единовременная оплата", r"единовременн|разов\w* (?:оплат|платеж)"),
        ("Верхний предел длительности", r"(?:не более|до)\s+14\b|максимальн.{0,110}\b14\b"),
        (
            "Календарные дни и начало отсчета",
            r"календарн.{0,16}дн.{0,35}(?:дат|момент).{0,18}подключен",
        ),
        ("Оба способа подключения", r"личн\w* кабинет\w* или (?:в |через )?мобильн\w* приложени"),
        ("Нет сроков соседней акции", r"^(?!.*(?:рассрочк|\b(?:6|12|24|25|26)\s*месяц))"),
    ],
    "07-07": [
        ("Базовый запрет", r"(?=.*(?:запрещ|запрет|не вправе))(?=.*субсубподрядчик)"),
        (
            "Совместные условия исключения",
            r"(?:исключени|кроме|только).{0,180}(?:согласован|согласия).{0,120}специализированн"
            r"|(?:исключени|кроме|только).{0,180}специализированн.{0,140}(?:согласован|согласия)",
        ),
        ("Штраф за каждый случай", r"(?=.*100\s?000\s+руб)(?=.*(?:за )?кажд\w* случа)"),
        ("Нет требования письменной формы", r"^(?!.*письменн)"),
    ],
    "05-03": [
        (
            "Хранение 14 дней с даты приостановления",
            r"14.{0,25}дн.{0,26}(?:дат|момент).{0,20}приостанов",
        ),
        (
            "Только данные с оборудования исполнителя",
            r"(?:если|при условии|при размещении).{0,85}оборудовани.{0,25}(?:исполнител|оператор)",
        ),
        (
            "Право безвозвратного удаления",
            r"(?:вправе|может).{0,15}удал.{0,120}(?:безвозврат|без возможности.{0,18}восстанов)",
        ),
        ("Автоматическое удаление", r"автоматическ"),
    ],
    "03-06": [
        (
            "Общий запрет перепродажи услуг",
            r"(?=.*(?:не имеет права|не вправе|запрещ|запрет))"
            r"(?=.*перепрода\w* (?:все\w* )?услуг\w* исполнителя)",
        ),
        ("Поддержка VDS, пароли и API", r"(?=.*поддержк\w* VDS)(?=.*парол)(?=.*API)"),
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, help="write the check results as JSON")
    args = parser.parse_args()
    run = args.work / "runs" / args.run
    response_path = run / "responses.jsonl"
    manifest = json.loads((run / "manifest.json").read_text())
    if digest(response_path) != manifest["responses_sha256"]:
        raise ValueError("saved responses differ from their manifest")
    rows = [json.loads(line) for line in response_path.read_text().splitlines()]
    results = []
    for case_id, criteria in CHECKS.items():
        selected = [r for r in rows if r["req_id"] == case_id]
        if len(selected) != 1:
            raise ValueError(f"{case_id}: expected one control response")
        row = selected[0]
        fid = row["target_fixture"]
        index_file = args.work / "index" / f"{fid}.json"
        if digest(index_file) != manifest["inputs"]["documents"][fid]["export_sha256"]:
            raise ValueError(f"{fid}: index export differs from the manifest")
        index = json.loads(index_file.read_text())
        if index["doc_id"] != row["doc_id"]:
            raise ValueError(f"{case_id}: request belongs to another document")
        chunks = {c["chunk_id"]: c for c in index["chunks"]}
        body = row["body"]
        text = re.sub(r"\s+", " ", body.get("answer", "")).replace("ё", "е")
        checks = {name: bool(re.search(pattern, text, re.I)) for name, pattern in criteria}
        citations = body.get("citations", [])
        exact = bool(citations)
        context = body.get("diagnostics", {}).get("context_chunk_ids", [])
        for q in citations:
            start, end = q["char_start"], q["char_end"]
            first, last = index["pages"][q["page"] - 1], index["pages"][q["page_to"] - 1]
            chunk = chunks.get(q["chunk_id"], {})
            exact &= (
                q["chunk_id"] in context
                and chunk.get("char_start", end) <= start < end <= chunk.get("char_end", start)
                and first["char_start"] <= start < first["char_end"]
                and last["char_start"] < end <= last["char_end"]
                and index["canonical_text"][start:end] == q["text"]
            )
        results.append(
            {
                "case_id": case_id,
                "http_status": row["http_status"],
                "found": body.get("found"),
                "checks": checks,
                "citations": len(citations),
                "all_exact": exact,
            }
        )
    report = {
        "run": args.run,
        "source_sha256": manifest["inputs"]["api_source_sha256"],
        "responses_sha256": manifest["responses_sha256"],
        "cases": results,
    }
    result = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")
    if not all(
        r["http_status"] == 200 and r["found"] and r["all_exact"] and all(r["checks"].values())
        for r in results
    ):
        raise SystemExit("contract control check failed")


if __name__ == "__main__":
    main()
