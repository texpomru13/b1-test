"""LLM-judged metrics for one QA run: RAGAS + an explicit fact/forbidden-fact check.

    cd eval && .venv/bin/python judge.py --run balanced-9b [--split all] [--limit N]

Judge: LOCAL Ollama model (default qwen3.6:27b-q4_K_M, thinking disabled,
temperature 0) — never the model under test, never a paid API. Safety rails:
  * RAGAS telemetry disabled (RAGAS_DO_NOT_TRACK) and OPENAI_API_KEY removed from
    the process, every metric gets the judge explicitly (no implicit default LLM).
RAGAS metrics (legacy single-turn API, ragas 0.4.3):
  faithfulness                  — response claims supported by the CONTEXT the
                                  model saw (found=true answers only)
  context_recall                — reference statements attributable to the context
  llm_context_precision_with_reference — ranking quality of the context chunks
  factual_correctness(F1)       — response vs reference claim overlap (found=true)
Fact check (one judge call per found=true answer): which expected facts are present,
whether any forbidden fact is asserted. Answerable+found=false => completeness 0.
Outputs eval/work/runs/<run>/judge_ragas.jsonl, judge_facts.jsonl, judge.json.
Resumable: rows already judged are skipped.
"""

from __future__ import annotations

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ.pop("OPENAI_API_KEY", None)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import ChatOllama  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"
MARKER = re.compile(r"\s*\[\d+\]")

FACT_PROMPT = """Ты — строгий проверяющий ответов по договору. Тебе дан вопрос, ответ системы,
список ОЖИДАЕМЫХ фактов и список ЗАПРЕЩЁННЫХ (неверных) фактов.
Для каждого ожидаемого факта реши, содержится ли он в ответе по смыслу (синонимы и
перефразировки допустимы; числа, сроки, проценты и условия должны совпадать).
Для каждого запрещённого факта реши, утверждает ли его ответ.
Верни ТОЛЬКО JSON вида {{"expected": [true/false, ...], "forbidden": [true/false, ...]}}
той же длины и в том же порядке, без пояснений.

Вопрос: {question}
Ответ системы: {answer}
Ожидаемые факты:
{expected}
Запрещённые факты:
{forbidden}
"""


FP_PROMPT = """Ты — строгий проверяющий. Вопрос задан по договору. Система утверждает, что
нашла ответ, и приводит дословные цитаты из договора. Реши ТОЛЬКО по цитатам:
отвечают ли они на вопрос по существу (а не на похожий, но другой вопрос)?
Верни ТОЛЬКО JSON {{"citations_answer_question": true/false, "reason": "кратко"}}.

Вопрос: {question}
Ответ системы: {answer}
Цитаты:
{quotes}
"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items)) or "(нет)"


def judge_facts(llm: ChatOllama, case: dict, answer: str) -> dict:
    prompt = FACT_PROMPT.format(question=case["question"], answer=answer,
                                expected=numbered(case["expected_facts"]),
                                forbidden=numbered(case["forbidden_facts"]))
    for attempt in range(3):
        raw = llm.invoke(prompt).content
        try:
            data = json.loads(raw)
            exp = [bool(x) for x in data["expected"]][: len(case["expected_facts"])]
            forb = [bool(x) for x in data.get("forbidden", [])][: len(case["forbidden_facts"])]
            if len(exp) == len(case["expected_facts"]):
                return {"expected": exp, "forbidden": forb, "attempts": attempt + 1}
        except (ValueError, KeyError, TypeError):
            continue
    return {"error": "judge output invalid after 3 attempts", "raw": str(raw)[:500]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    ap.add_argument("--judge", default="qwen3.6:27b-q4_K_M")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ragas", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    gold = {c["case_id"]: c for c in load_jsonl(ROOT / "eval" / "gold" / "gold.jsonl")}
    index = {json.loads(p.read_text(encoding="utf-8"))["doc_id"]:
             json.loads(p.read_text(encoding="utf-8")) for p in (WORK / "index").glob("*.json")}
    chunk_text = {d: {c["chunk_id"]: c["text"] for c in v["chunks"]} for d, v in index.items()}
    run_dir = WORK / "runs" / args.run
    recs = [r for r in load_jsonl(run_dir / "responses.jsonl")
            if r["kind"] == "gold" and r.get("http_status") == 200
            and gold[r["case_id"]]["answerable"]
            and (args.split == "all" or gold[r["case_id"]]["split"] == args.split)]
    if args.limit:
        recs = recs[: args.limit]

    chat = ChatOllama(model=args.judge, base_url=args.ollama, temperature=0, seed=7,
                      reasoning=False, num_ctx=16384, num_predict=2048)
    chat_json = ChatOllama(model=args.judge, base_url=args.ollama, temperature=0, seed=7,
                           reasoning=False, num_ctx=8192, num_predict=512, format="json")

    # ---- explicit fact check -------------------------------------------------
    facts_path = run_dir / "judge_facts.jsonl"
    done = {r["req_id"] for r in load_jsonl(facts_path)}
    t0 = time.monotonic()
    with facts_path.open("a", encoding="utf-8") as fh:
        for rec in recs:
            if rec["req_id"] in done:
                continue
            case, body = gold[rec["case_id"]], rec["body"]
            row = {"req_id": rec["req_id"], "case_id": case["case_id"],
                   "found": body.get("found")}
            if body.get("found"):
                row.update(judge_facts(chat_json, case, MARKER.sub("", body["answer"])))
            else:
                row.update({"expected": [False] * len(case["expected_facts"]),
                            "forbidden": [False] * len(case["forbidden_facts"]),
                            "note": "found=false on an answerable case"})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"fact check done in {time.monotonic() - t0:.0f}s", flush=True)

    # ---- false positives on negatives / isolation probes --------------------------
    # A found=true on a no-answer case (or on a foreign document) is either a
    # system hallucination (quotes do not answer) or a gold error (the document
    # does answer). The judge decides from the verified quotes; the report lists
    # every case for manual review.
    fp_path = run_dir / "judge_false_positives.jsonl"
    done_fp = {r["req_id"] for r in load_jsonl(fp_path)}
    fps = [r for r in load_jsonl(run_dir / "responses.jsonl")
           if r.get("http_status") == 200 and r["body"].get("found")
           and (r["kind"] == "isolation" or not gold[r["case_id"]]["answerable"])
           and (args.split == "all" or gold[r["case_id"]]["split"] == args.split)]
    with fp_path.open("a", encoding="utf-8") as fh:
        for rec in fps:
            if rec["req_id"] in done_fp:
                continue
            quotes = "\n".join(f"[{c['id']}] {c['text']}" for c in rec["body"]["citations"])
            prompt = FP_PROMPT.format(question=rec["question"],
                                      answer=MARKER.sub("", rec["body"]["answer"]),
                                      quotes=quotes)
            verdict: dict = {"error": "judge output invalid"}
            for _ in range(3):
                try:
                    data = json.loads(chat_json.invoke(prompt).content)
                    verdict = {"citations_answer_question":
                               bool(data["citations_answer_question"]),
                               "reason": str(data.get("reason", ""))[:300]}
                    break
                except (ValueError, KeyError, TypeError):
                    continue
            fh.write(json.dumps({"req_id": rec["req_id"], "kind": rec["kind"],
                                 "case_id": rec["case_id"], "question": rec["question"],
                                 "answer": rec["body"]["answer"], **verdict},
                                ensure_ascii=False) + "\n")

    # ---- RAGAS -----------------------------------------------------------------
    ragas_path = run_dir / "judge_ragas.jsonl"
    if not args.skip_ragas:
        done_r = {r["req_id"] for r in load_jsonl(ragas_path)}
        todo = [r for r in recs if r["req_id"] not in done_r]
        llm = LangchainLLMWrapper(chat)
        cfg = RunConfig(timeout=1800, max_retries=3, max_workers=args.workers)
        metrics_found = [Faithfulness(llm=llm), LLMContextRecall(llm=llm),
                         LLMContextPrecisionWithReference(llm=llm),
                         FactualCorrectness(llm=llm, mode="f1")]
        metrics_nf = [LLMContextRecall(llm=llm), LLMContextPrecisionWithReference(llm=llm)]
        for m in metrics_found + metrics_nf:
            assert m.llm is llm, "every RAGAS metric must use the local judge"
        batch = 8
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            for found_flag, metrics in ((True, metrics_found), (False, metrics_nf)):
                sel = [r for r in part if bool(r["body"].get("found")) == found_flag]
                if not sel:
                    continue
                samples = []
                for r in sel:
                    case = gold[r["case_id"]]
                    ctx_ids = (r["body"].get("diagnostics") or {}).get("context_chunk_ids", [])
                    contexts = [chunk_text[r["doc_id"]][c] for c in ctx_ids
                                if c in chunk_text[r["doc_id"]]] or ["(пустой контекст)"]
                    samples.append(SingleTurnSample(
                        user_input=case["question"],
                        response=MARKER.sub("", r["body"]["answer"]) if found_flag else None,
                        retrieved_contexts=contexts,
                        reference=case["reference_answer"],
                    ))
                t1 = time.monotonic()
                result = evaluate(EvaluationDataset(samples=samples), metrics=metrics,
                                  run_config=cfg, show_progress=False)
                scores = result.scores
                with ragas_path.open("a", encoding="utf-8") as fh:
                    for r, sc in zip(sel, scores, strict=True):
                        fh.write(json.dumps({"req_id": r["req_id"], "case_id": r["case_id"],
                                             "found": found_flag,
                                             **{k: (None if v != v else v) for k, v in sc.items()}},
                                            ensure_ascii=False) + "\n")
                print(f"ragas batch {i // batch + 1} found={found_flag} n={len(sel)} "
                      f"in {time.monotonic() - t1:.0f}s", flush=True)

    # ---- aggregate --------------------------------------------------------------
    # aggregate ONLY the rows of this split (the jsonl files are shared across splits)
    in_split = {r["req_id"] for r in recs}
    facts = [f for f in load_jsonl(facts_path) if f["req_id"] in in_split]
    ragas_rows = [r for r in load_jsonl(ragas_path) if r["req_id"] in in_split]
    fp_rows = [r for r in load_jsonl(fp_path)
               if args.split == "all" or gold[r["case_id"]]["split"] == args.split]
    exp_tot = sum(len(f.get("expected", [])) for f in facts if "error" not in f)
    exp_hit = sum(sum(f.get("expected", [])) for f in facts if "error" not in f)
    complete = sum(1 for f in facts if "error" not in f and f["expected"] and all(f["expected"]))
    forb_cases = sum(1 for f in facts if any(f.get("forbidden", [])))
    found_facts = [f for f in facts if f.get("found") and "error" not in f]
    summary = {
        "judge": args.judge, "split": args.split,
        "fact_recall_all_answerable": {"num": exp_hit, "den": exp_tot,
                                       "value": round(exp_hit / exp_tot, 4) if exp_tot else None},
        "complete_answers_all_answerable": {"num": complete, "den": len(facts)},
        "fact_recall_found_only": {
            "num": sum(sum(f["expected"]) for f in found_facts),
            "den": sum(len(f["expected"]) for f in found_facts)},
        "answers_with_forbidden_fact": {"num": forb_cases, "den": len(facts)},
        "judge_errors": sum(1 for f in facts if "error" in f),
        "false_positives": {
            "n": len(fp_rows),
            "quotes_do_not_answer (hallucination)": sum(
                1 for r in fp_rows if r.get("citations_answer_question") is False),
            "quotes_answer (possible gold error)": sum(
                1 for r in fp_rows if r.get("citations_answer_question") is True),
        },
        "ragas": {},
    }
    for key in ("faithfulness", "context_recall", "llm_context_precision_with_reference",
                "factual_correctness(mode=f1)"):
        for scope, rows in (("found", [r for r in ragas_rows if r["found"]]),
                            ("all", ragas_rows)):
            vals = [r[key] for r in rows if r.get(key) is not None]
            missing = sum(1 for r in rows if key in r and r[key] is None)
            if vals or missing:
                summary["ragas"][f"{key}[{scope}]"] = {
                    "mean": round(statistics.fmean(vals), 4) if vals else None,
                    "n": len(vals), "nan": missing}
    (run_dir / f"judge-{args.split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
