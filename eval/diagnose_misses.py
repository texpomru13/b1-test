"""Diagnose answerable cases that ended found=false: at which QA step did the repeated request stop?

    uv run python eval/diagnose_misses.py --review-submission
    uv run python eval/diagnose_misses.py --run balanced-9b-nothink [--project b1-eval]

Re-asks each missed answerable case once (debug=true) and collects the API log
events emitted during that request (draft_invalid / provenance_rejected /
verifier_invalid / question_answered). Diagnosis only: nothing is tuned on these
cases. Writes eval/work/runs/<run>/miss_diagnosis.jsonl and prints a breakdown.
Note: a re-ask is a new sample (the model is not deterministic), so the
outcome may differ from the original run; both are recorded.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import time
from pathlib import Path

import httpx
from paths import ROOT, WORK

# Per-case adjudication of the archived request. Keep the acceptance condition
# explicit: a matching span may still fail to support the scope of a claim.
_REVIEW = {
    "01-02": "Указать пару ИНН/КПП и ОГРН именно покупателя. Цифра в названии организации "
             "требует подтверждения полного названия в том же блоке реквизитов.",
    "01-03": "Применить правило для срока от 18 месяцев. Остаток при приёмке — минимум год. "
             "Два года заданы вопросом. Преобразование в 24 месяца относится к примеру.",
    "03-05": "Подтвердить письменное уведомление за 30 календарных дней и возврат "
             "только за полные неиспользованные месяцы.",
    "05-03": "Передать срок 14 дней, условие размещения на оборудовании исполнителя "
             "и его право удалить сведения без восстановления по истечении срока.",
    "10-01": "Связать расходы по е) и ж) с пунктом 5.1.1, обе базы для 10% — "
             "со своими лимитами, сохранить оговорку об ином условии договора.",
    "10-04": "Указать обязанность сообщить незамедлительно, крайний срок три календарных "
             "дня и исключение для выходных и праздничных дней.",
    "11-06": "Два самостоятельных вывода требуют доказательств. Пропуск занятий "
             "не даёт уменьшения платы и права ссылаться на неоказание услуг.",
    "13-03": "При отсутствии пункта об оплате сохранить отказ. Для ответа требуется "
             "подтвердить перечисление средств на банковский счёт исполнителя.",
    "13-05": "Различить соглашение сторон и одностороннее расторжение по основаниям закона. "
             "Указать письменное уведомление и незаполненный срок без вымышленной даты.",
    "14-01": "Связать цену 79 руб./мес. и состав пакета с тарифом «1 день Полная запись» "
             "в одной строке или ячейках этой строки.",
    "14-02": "Связать цену 3650 руб. за 30 дней, до 10 камер и FullHD с одной подпиской "
             "«Профи». Нужны подпись строки и соответствующие столбцы.",
    "14-04": "Сохранить отказ до появления применимого гарантийного пункта. Для ответа "
             "нужны 24 месяца и момент приобретения оборудования.",
    "14-08": "В показанных фрагментах отсутствует правило о количестве камер. Нужны лимит "
             "3 единицы и исключение до 1 для личной явки представителя службы продаж. "
             "Однократность тестового периода не определяет количество камер.",
}


def review_submission(directory: Path) -> str:
    """Cross-check the published diagnostic events against HTTP results and annotation."""
    def load_rows(path: Path, key: str) -> dict[str, dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        result = {row[key]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"duplicate {key} in {path}")
        return result

    gold = load_rows(ROOT / "eval/gold/gold.jsonl", "case_id")
    responses = load_rows(directory / "responses.jsonl", "req_id")
    failed = {rid for rid, r in responses.items() if r["kind"] == "gold"
              and gold[r["case_id"]]["answerable"] and r["http_status"] == 200
              and r["body"]["found"] is False}
    reports = json.loads((directory / "refusal-analysis.json").read_text())["cases"]
    if (failed != set(_REVIEW) or failed != {r["case_id"] for r in reports}
            or len(reports) != len(failed)):
        raise ValueError("review must cover exactly the observed answerable refusals")
    lines = ["# Диагностика отказов по журналу HTTP-запросов", "",
             "В выборку вошли 13 вопросов, на которые в договоре есть ответ. "
             "Сервис вернул HTTP 200 и `found=false`. Столбец «Выдержек в контексте» "
             "показывает, сколько эталонных выдержек попало в контекст модели хотя бы "
             "на 80% длины. В десяти контекстах есть все эталонные выдержки. В трёх "
             "искомый пункт отсутствует. [Ответы API](submission/responses.jsonl) · "
             "[События обработки](submission/refusal-analysis.json) · "
             "[Вопросы и эталоны](../gold/gold.jsonl)", "",
             "| Вопрос | Этап остановки | Выдержек в контексте | Условие корректного ответа |",
             "|---|---|---|---|"]
    for row in sorted(reports, key=lambda r: r["case_id"]):
        cid = row["case_id"]
        r = responses[cid]
        if (r["request_id"] != row["request_id"] or r["question"] != gold[cid]["question"]
                or not row["events"]
                or any(e["request_id"] != r["request_id"] for e in row["events"])
                or row["gold_span_count"] != len(gold[cid]["gold_spans"])):
            raise ValueError(f"{cid}: archived inputs disagree")
        count = f'{len(row["context_gold_spans"])}/{row["gold_span_count"]}'
        lines.append(f'| {cid} | `{row["final_classification"]}` | {count} | {_REVIEW[cid]} |')
    lines.extend(["", "В случае 14-08 запись журнала позволяет определить этап остановки "
                  "косвенно, после успешного вердикта модели. Для остальных строк журнал "
                  "содержит явное решение. Смысл ответов при подаче этих же фрагментов "
                  "проверен в отдельном эксперименте. [Генерация с фиксированным "
                  "контекстом](refusal-replay/README.md)", "",
                  "Сверку событий, вопросов и отказов воспроизводит команда "
                  "`.venv/bin/python eval/diagnose_misses.py --review-submission`.", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run")
    mode.add_argument("--review-submission", action="store_true")
    ap.add_argument("--project", default="b1-eval")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    if args.review_submission:
        print(review_submission(ROOT / "eval/reports/submission"), end="")
        return
    run_dir = WORK / "runs" / args.run
    cases = []
    for split in ("dev", "holdout"):
        rows = [json.loads(x) for x in (run_dir / f"cases-{split}.jsonl").read_text().splitlines()]
        cases += [r for r in rows if r["kind"] == "gold" and r["answerable"]
                  and r["outcome"] == "not_found"]
    responses = {json.loads(x)["req_id"]: json.loads(x)
                 for x in (run_dir / "responses.jsonl").read_text().splitlines() if x}
    out_path = run_dir / "miss_diagnosis.jsonl"
    client = httpx.Client(base_url=args.api, timeout=900)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in cases:
            rec = responses[row["req_id"]]
            # unix seconds: an RFC3339 string without a zone is read as LOCAL time
            since = str(int(time.time()) - 1)
            resp = client.post(f"/api/v1/documents/{rec['doc_id']}/questions",
                               json={"question": rec["question"], "debug": True})
            request_id = resp.headers["X-Request-ID"]
            time.sleep(1.0)
            # structlog writes to the container's stderr: merge both streams
            logs = subprocess.run(["docker", "logs", "--since", since, f"{args.project}-api-1"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True).stdout.splitlines()
            events = []
            for line in logs:
                if line.startswith("{"):
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("request_id") != request_id:
                        continue
                    if e.get("event") in ("draft_invalid", "provenance_rejected",
                                          "verifier_invalid", "question_answered",
                                          "llm_outcome_uncertain", "verifier_verdict",
                                          "draft_not_answerable"):
                        events.append({k: e.get(k) for k in ("event", "error", "found",
                                                             "llm_calls", "repairs", "coverage",
                                                             "verdicts") if k in e})
            body = resp.json()
            usage = (body.get("diagnostics") or {}).get("token_usage", {})
            final = next((e for e in events if e["event"] == "question_answered"), {})
            decisions = [e for e in events if e["event"] in (
                "draft_not_answerable", "provenance_rejected", "draft_invalid", "verifier_invalid"
            )]
            if resp.status_code != 200:
                stage = f"http_{resp.status_code}"
            elif body.get("found"):
                stage = "answered_on_reask"
            elif decisions:
                last = decisions[-1]
                stage = last.get("error") or last["event"]
            else:
                stage = "unclassified"
            fh.write(json.dumps({"req_id": row["req_id"], "request_id": request_id,
                                 "category": row["category"],
                                 "ctx_all": row.get("ctx_all"), "reask_http": resp.status_code,
                                 "reask_found": body.get("found"),
                                 "llm_calls": usage.get("llm_calls"),
                                 "repairs": final.get("repairs"), "events": events, "stage": stage},
                                ensure_ascii=False) + "\n")
            print(row["req_id"], stage, flush=True)
    rows = [json.loads(x) for x in out_path.read_text().splitlines()]
    print(json.dumps(collections.Counter(r["stage"] for r in rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
