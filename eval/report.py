"""Build a Markdown report from measurements in the evaluation work directory.

    python3 eval/report.py --runs balanced-check
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from paths import WORK, gold_path

TARGETS = {
    "citation_span_validity": ("Валидность цитат (точная подстрока)", "100%", 1.0),
    "cross_doc_leakage_citations": ("Утечка цитат из другого документа", "0", 0.0),
    "positive_hit_at_candidates": ("Доказательство среди кандидатов (hit@60)", ">=95%", 0.95),
    "context_evidence_coverage_all": ("Все доказательства в контексте LLM", ">=90%", 0.90),
    "correct_abstention_negatives": ("Верный отказ на вопросах без ответа", ">=95%", 0.95),
    "structured_validity_200": ("Валидность схемы ответов 200", "100%", 1.0),
}
VERDICT = {True: "выполнено", False: "не выполнено"}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fmt(r: dict | None) -> str:
    if not r or r.get("den") in (None, 0):
        return "н/д (0 случаев)" if r else "not_run"
    return f"{r['num']}/{r['den']} = {100 * r['value']:.1f}%"


def verdict(key: str, r: dict | None) -> str:
    if not r or r.get("value") is None:
        return "not_run"
    target = TARGETS[key][2]
    value = r["num"] / r["den"]
    ok = value <= target if key == "cross_doc_leakage_citations" else value >= target
    return VERDICT[ok]


def run_section(name: str) -> list[str]:
    lines = [f"### Прогон `{name}`", ""]
    meta = load(WORK / "runs" / name / "meta.json")
    if meta is None:
        return [*lines, "not_run", ""]
    lines.append(f"Начало измерений — {meta['started_at']}. Адрес API — {meta['api']}.")
    lines.append("")
    for split in ("all", "holdout", "dev"):
        m = load(WORK / "runs" / name / f"metrics-{split}.json")
        if m is None:
            continue
        lines += [f"**Сплит `{split}`** — запросов {m['n_requests']} (gold {m['n_gold']}: "
                  f"с ответом {m['n_answerable']}, без ответа {m['n_no_answer']}; "
                  f"проб изоляции {m['n_isolation']})", ""]
        partial = bool(m.get("missing_gold"))
        if partial:
            lines += [f"Частичный прогон: собрано {m['n_gold']} из "
                      f"{m['n_gold_planned']} вопросов эталона.", ""]
        lines += ["| Метрика | Цель | Измерено | Итог |", "|---|---|---|---|"]
        for key, (label, target, _) in TARGETS.items():
            result = "частичный прогон" if partial else verdict(key, m.get(key))
            lines.append(f"| {label} | {target} | {fmt(m.get(key))} | {result} |")
        lines += [
            f"| Пробы изоляции → found=false | 100% | "
            f"{fmt(m.get('isolation_correct_not_found'))} | — |",
            f"| Вопрос с ответом → found=true | — | {fmt(m.get('answerable_found'))} | — |",
            f"| Вопрос с ответом → found=false (промах) | — | "
            f"{fmt(m.get('answerable_not_found'))} | — |",
            f"| Вопрос с ответом → ошибка HTTP, транспорта или схемы | 0 | "
            f"{fmt(m.get('answerable_errors'))} | — |",
            f"| Вопрос без ответа → ошибка | 0 | {fmt(m.get('negatives_errors'))} | — |",
            f"| found=true с цитатой из gold-фрагмента | — | "
            f"{fmt(m.get('found_cites_gold_span'))} | — |",
            f"| Выданные ответы, в которых цитаты покрывают все эталонные выдержки | — | "
            f"{fmt(m.get('found_covers_all_gold_spans'))} | — |",
            f"| Средняя доля gold-фрагментов в контексте | — | "
            f"{m.get('mean_span_recall_context')} | — |",
            f"| nDCG@10 (после реранкера) | — | {m.get('ndcg@10_rerank')} | — |",
            "",
        ]
        cats = m.get("by_category", {})
        if cats:
            lines += ["По категориям (с ответом: found=true; no_answer: found=false):", "",
                      "| Категория | Верный исход |", "|---|---|"]
            lines += [f"| {c} | {fmt(v)} |" for c, v in cats.items()]
            lines.append("")
        lat = m.get("latency_ms_200", {})
        if lat.get("n"):
            lines += [f"Задержка ответов 200 (время клиента, запросы последовательно): "
                      f"n={lat['n']}, p50={lat['p50'] / 1000:.1f} с, "
                      f"p95={lat['p95'] / 1000:.1f} с, max={lat['max'] / 1000:.1f} с. "
                      f"p50 по стадиям, мс: {m.get('stage_ms_p50')}.", ""]
        j = load(WORK / "runs" / name / f"judge-{split}.json")
        if j:
            if coverage := j.get("coverage"):
                lines += [f"Судье доступны {coverage['valid_answerable']} из "
                          f"{coverage['attempted_answerable']} записанных положительных запросов. "
                          f"Ошибок HTTP, транспорта или схемы: {coverage['failed_responses']}.", ""]
            ff = j["fact_recall_found_only"]
            ff_val = dict(ff, value=(ff["num"] / ff["den"]) if ff["den"] else None)
            lines += [f"Оценка судьёй (`{j['judge']}`, локально):", "",
                      "| Метрика | Значение |", "|---|---|",
                      f"| Полнота фактов по оценённым ответам на положительные вопросы | "
                      f"{fmt(j['fact_recall_all_answerable'])} |",
                      f"| Полнота ожидаемых фактов, только found=true | {fmt(ff_val)} |",
                      f"| Полные ответы (все факты) | "
                      f"{j['complete_answers_all_answerable']['num']}/"
                      f"{j['complete_answers_all_answerable']['den']} |",
                      f"| Ответы с запрещённым фактом | "
                      f"{j['answers_with_forbidden_fact']['num']}/"
                      f"{j['answers_with_forbidden_fact']['den']} |",
                      f"| Ошибки вывода судьи | {j['judge_errors']} |"]
            fp = j.get("false_positives")
            if fp:
                lines.append(f"| found=true на вопросе без ответа / пробе изоляции: "
                             f"галлюцинация по судье | "
                             f"{fp['quotes_do_not_answer (hallucination)']}/{fp['n']} |")
                lines.append(f"| found=true на вопросе без ответа / пробе изоляции: цитаты "
                             f"отвечают (ошибка gold?) | "
                             f"{fp['quotes_answer (possible gold error)']}/{fp['n']} |")
            for k, v in j.get("ragas", {}).items():
                mean = "н/д" if v["mean"] is None else f"{v['mean']:.3f}"
                lines.append(f"| RAGAS {k} | {mean} (n={v['n']}, NaN={v['nan']}) |")
            lines.append("")
        fp_path = WORK / "runs" / name / "judge_false_positives.jsonl"
        saved = (json.loads(x) for x in fp_path.read_text(encoding="utf-8").splitlines() if x
                 ) if fp_path.exists() else ()
        fps = list({row["req_id"]: row for row in saved}.values())
        if fps and split == "all":
            lines += ["Ложные срабатывания (found=true там, где ожидался отказ) — все случаи:",
                      "", "| Запрос | Судья: цитаты отвечают? | Обоснование судьи |",
                      "|---|---|---|"]
            for r in fps:
                reason = " ".join(r.get("reason", r.get("error", "")).split()).replace("|", "/")
                lines.append(f"| {r['req_id']} | {r.get('citations_answer_question')} | "
                             f"{reason[:160]} |")
            lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", type=Path, default=WORK / "REPORT.md")
    args = ap.parse_args()
    runs = [r for r in args.runs.split(",") if r]
    gold = [json.loads(x) for x in gold_path().read_text().splitlines() if x]
    lines = [
        "# Оценка ответов по документам", "",
        f"Дата отчёта — {datetime.now(UTC).isoformat(timespec='seconds')}.", "",
        f"Эталон содержит {len(gold)} вопросов по "
        f"{len({c['doc_fixture_id'] for c in gold})} документам "
        f"(dev {sum(c['split'] == 'dev' for c in gold)}, "
        f"holdout {sum(c['split'] == 'holdout' for c in gold)}). Вопросы составлены моделями, "
        "цитаты сверены с каноническим текстом автоматически. Независимой экспертной разметки нет.",
        "",
    ]
    for r in runs:
        lines += run_section(r)
    abl = load(WORK / "ablation" / "balanced.metrics.json")
    if abl:
        lines += ["## Сравнение способов поиска на одном индексе", ""]
        keys = ["recall@1", "recall@5", "recall@8", "recall@20", "recall@60", "hit@60",
                "all_spans@8", "ndcg@10", "mrr"]
        for split in ("all", "holdout"):
            lines += [f"Сплит `{split}` (n={abl[split]['dense']['n']} вопросов с ответом):", "",
                      "| Метод | " + " | ".join(keys) + " |",
                      "|---|" + "---|" * len(keys)]
            for meth, vals in abl[split].items():
                lines.append(f"| {meth} | " + " | ".join(str(vals.get(k)) for k in keys) + " |")
            lines.append("")
    comp = load(WORK / "compare" / "report.json")
    if comp:
        lines += ["## Сравнение моделей на одних фрагментах и кандидатах", ""]
        keys = ["recall@1", "recall@5", "recall@8", "recall@20", "recall@60", "all_spans@8",
                "ndcg@10"]
        lines += ["| Энкодер | Режим | " + " | ".join(keys) + " | кодирование, с |",
                  "|---|---|" + "---|" * (len(keys) + 1)]
        for enc, e in comp["encoders"].items():
            for mode in ("dense[all]", "hybrid_rrf60[all]", "dense[holdout]",
                         "hybrid_rrf60[holdout]"):
                lines.append(f"| {enc} | {mode} | " + " | ".join(str(e[mode].get(k))
                                                               for k in keys)
                             + f" | {e['encode_s']} |")
        lines.append("")
        lines += ["| Реранкер | Кандидаты от | " + " | ".join(keys) + " | оценка, с |",
                  "|---|---|" + "---|" * (len(keys) + 1)]
        for rr, e in comp["rerankers"].items():
            for k2, v in e.items():
                if isinstance(v, dict):
                    lines.append(f"| {rr} | {k2} | " + " | ".join(str(v.get(k)) for k in keys)
                                 + f" | {e['score_s']} |")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
