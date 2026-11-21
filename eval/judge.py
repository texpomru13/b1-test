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

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import ChatOllama  # noqa: E402
from paths import ROOT, WORK, gold_path  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig  # noqa: E402

sys.path.insert(0, str(ROOT))
from app.schemas.qa import AnswerResponse  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def latest_rows(path: Path) -> dict[str, dict]:
    return {row["req_id"]: row for row in load_jsonl(path)}


def verdict_list(value: list, size: int) -> list[bool]:
    if not isinstance(value, list) or len(value) != size or any(type(v) is not bool for v in value):
        raise ValueError("judge must return one JSON boolean per fact")
    return value


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
            exp = verdict_list(data["expected"], len(case["expected_facts"]))
            forb = verdict_list(data["forbidden"], len(case["forbidden_facts"]))
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
    ap.add_argument("--context-tokens", type=int, default=8192)
    args = ap.parse_args()

    gold = {c["case_id"]: c for c in load_jsonl(gold_path())}
    index = {json.loads(p.read_text(encoding="utf-8"))["doc_id"]:
             json.loads(p.read_text(encoding="utf-8")) for p in (WORK / "index").glob("*.json")}
    chunk_text = {d: {c["chunk_id"]: c["text"] for c in v["chunks"]} for d, v in index.items()}
    run_dir = WORK / "runs" / args.run
    response_file = run_dir / "responses.jsonl"
    if not response_file.is_file():
        ap.error(f"missing responses: {response_file}")
    selected = [r for r in load_jsonl(response_file)
                if args.split in ("all", gold[r["case_id"]]["split"])]
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    if args.limit:
        selected = selected[:args.limit]
    if len({r["req_id"] for r in selected}) != len(selected):
        ap.error("duplicate req_id in responses.jsonl")
    valid = []
    for r in selected:
        if r.get("http_status") == 200:
            try:
                AnswerResponse.model_validate(r.get("body"), strict=True)
            except ValidationError:
                continue
            valid.append(r)
    positives = [r for r in selected if r["kind"] == "gold" and gold[r["case_id"]]["answerable"]]
    recs = [r for r in valid if r["kind"] == "gold" and gold[r["case_id"]]["answerable"]]
    judge_meta = run_dir / "judge_model.json"
    identities = {s["judge"] for path in run_dir.glob("judge-*.json")
                  if (s := json.loads(path.read_text())).get("judge")}
    if judge_meta.exists():
        identities.add(json.loads(judge_meta.read_text())["judge"])
    if identities - {args.judge}:
        ap.error(f"saved verdicts use a different judge {sorted(identities)}; use a new --run")
    judge_meta.write_text(json.dumps({"judge": args.judge}), encoding="utf-8")

    chat = ChatOllama(model=args.judge, base_url=args.ollama, temperature=0, seed=7,
                      reasoning=False, num_ctx=16384, num_predict=2048)
    chat_json = ChatOllama(model=args.judge, base_url=args.ollama, temperature=0, seed=7,
                           reasoning=False, num_ctx=args.context_tokens,
                           num_predict=512, format="json")

    # ---- explicit fact check -------------------------------------------------
    facts_path = run_dir / "judge_facts.jsonl"
    saved_facts = latest_rows(facts_path)
    t0 = time.monotonic()
    with facts_path.open("a", encoding="utf-8") as fh:
        for rec in recs:
            case, body = gold[rec["case_id"]], rec["body"]
            # A changed rubric or answer needs a new verdict even for the same request ID.
            judge_input = [FACT_PROMPT, case["question"], case["expected_facts"],
                           case["forbidden_facts"], body["found"], body["answer"]]
            encoded = json.dumps(judge_input, ensure_ascii=False).encode()
            input_hash = hashlib.sha256(encoded).hexdigest()
            saved = saved_facts.get(rec["req_id"], {})
            if saved.get("input_sha256") == input_hash and "error" not in saved:
                continue
            row = {"req_id": rec["req_id"], "case_id": case["case_id"],
                   "found": body["found"], "input_sha256": input_hash}
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
    done_fp = {k for k, r in latest_rows(fp_path).items() if "error" not in r}
    fps = [r for r in valid if r["body"]["found"]
           and (r["kind"] == "isolation" or not gold[r["case_id"]]["answerable"])]
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
                    if type(data["citations_answer_question"]) is not bool:
                        raise ValueError("expected a JSON boolean")
                    verdict = {"citations_answer_question": data["citations_answer_question"],
                               "reason": str(data.get("reason", ""))[:300]}
                    break
                except (ValueError, KeyError, TypeError):
                    continue
            fh.write(json.dumps({"req_id": rec["req_id"], "kind": rec["kind"],
                                 "case_id": rec["case_id"], "question": rec["question"],
                                 "answer": rec["body"]["answer"], **verdict},
                                ensure_ascii=False) + "\n")
            fh.flush()

    # ---- RAGAS -----------------------------------------------------------------
    ragas_path = run_dir / "judge_ragas.jsonl"
    if not args.skip_ragas:
        done_r = {k for k, r in latest_rows(ragas_path).items() if None not in r.values()}
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
                    contexts = [chunk_text[r["doc_id"]][c] for c in ctx_ids]
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
    facts = [f for k, f in latest_rows(facts_path).items() if k in in_split]
    ragas_rows = [r for k, r in latest_rows(ragas_path).items() if k in in_split]
    fp_ids = {r["req_id"] for r in fps}
    fp_rows = [r for k, r in latest_rows(fp_path).items() if k in fp_ids]
    valid_facts = [f for f in facts if "error" not in f]
    exp_tot = sum(len(f.get("expected", [])) for f in facts if "error" not in f)
    exp_hit = sum(sum(f.get("expected", [])) for f in facts if "error" not in f)
    complete = sum(1 for f in facts if "error" not in f and f["expected"] and all(f["expected"]))
    forb_cases = sum(1 for f in facts if any(f.get("forbidden", [])))
    found_facts = [f for f in facts if f.get("found") and "error" not in f]
    summary = {
        "judge": args.judge, "split": args.split,
        "gold_sha256": hashlib.sha256(gold_path().read_bytes()).hexdigest(),
        "coverage": {"attempted_answerable": len(positives), "valid_answerable": len(recs),
                     "failed_responses": len(positives) - len(recs), "judged": len(valid_facts)},

        "fact_recall_all_answerable": {"num": exp_hit, "den": exp_tot,
                                       "value": round(exp_hit / exp_tot, 4) if exp_tot else None},
        "complete_answers_all_answerable": {"num": complete, "den": len(valid_facts)},
        "fact_recall_found_only": {
            "num": sum(sum(f["expected"]) for f in found_facts),
            "den": sum(len(f["expected"]) for f in found_facts)},
        "answers_with_forbidden_fact": {"num": forb_cases, "den": len(valid_facts)},
        "judge_errors": sum(1 for f in facts if "error" in f),
        "false_positives": {
            "n": sum("error" not in r for r in fp_rows),
            "errors": sum("error" in r for r in fp_rows),
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
    suffix = f"-limit{args.limit}" if args.limit else ""
    (run_dir / f"judge-{args.split}{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
