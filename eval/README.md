# Evaluation (stage 9)

Reproducible evaluation of the RAG service against a gold set built from
`test_documents/`. The headline report is `eval/reports/REPORT.md`; raw measured
artifacts of the reported runs are in `eval/reports/runs/`.

Nothing here calls a paid API: the system under test uses the local LLM, and the
LLM judge is a local Ollama model (`qwen3.6:27b-q4_K_M`, thinking off, temperature 0).

## Layout

| Path | What |
|---|---|
| `gold/gold.jsonl` | Gold cases: question, category, answerable, reference answer, expected/forbidden facts, gold spans (exact quotes with recovered offsets/pages), split |
| `gold/drafts/*.jsonl` | Per-document drafts the gold set is built from (`build_gold.py`) |
| `gold/DRAFTING_BRIEF.md` | Rules the drafts were written under |
| `gold/manifest.json` | Corpus manifest: SHA-256, pages, canonical text hash, chunk counts, origin |
| `upload_corpus.py` | Uploads the corpus via the API, times background indexing (+ a 100-page fixture) |
| `export_index.py` | Exports canonical text + published chunks from the running stack (read-only) |
| `build_gold.py` | Validates drafts: every quote must be an exact substring of the canonical text |
| `run_qa.py` | Asks every gold question (+ cross-document isolation probes) via the public API, `debug=true` |
| `metrics.py` | Deterministic metrics: schema/citation validity, leakage, retrieval hit/coverage, abstention, latency |
| `judge.py` | LLM-judged metrics: RAGAS (faithfulness, context recall/precision, factual correctness) + fact check |
| `ablation.py` | dense / BM25 / hybrid RRF / hybrid DBSF / hybrid+rerank on the same gold spans |
| `compare_models.py` | Encoders and rerankers on fixed chunk boundaries and frozen candidates |
| `report.py` | Builds `reports/REPORT.md` from the artifacts |
| `demo.sh` | API demo (upload, poll, list, exact, paraphrase, no-answer, debug, isolation) with saved outputs |

## Gold set

133 cases over 14 text documents (document 06 is a scan without a text layer and
is used only as the `OCR_REQUIRED` ingestion case). Categories: exact 30,
paraphrase 29, multi-condition 16, negation/limits 18, no-answer 40. The split is
by document: dev = fixtures 01, 05, 13, 15 (35 cases), holdout = the other 10
(98 cases). **Provenance:** drafted by LLM agents from the canonical text; every
quote is machine-verified as an exact substring; absence of no-answer facts was
checked by text search (recorded in `notes`). The set has NOT been reviewed by a
human lawyer — treat it as a strong smoke set, not a legal ground truth.

## Reproduce

```bash
# 1. an isolated stack (own volumes; does not touch the default b1-rag project)
B1_PROJECT=b1-eval ./run --quality balanced
# 2. corpus, index export, gold
uv run python eval/upload_corpus.py
python3 eval/export_index.py --project b1-eval
python3 eval/build_gold.py
# 3. retrieval ablation and the QA run
python3 eval/ablation.py --project b1-eval --name balanced
uv run python eval/run_qa.py --run balanced-9b
uv run python eval/metrics.py --run balanced-9b --split holdout   # also: all, dev
# 4. LLM judge (separate env: cd eval && uv sync)
cd eval && .venv/bin/python judge.py --run balanced-9b --split all && cd ..
# 5. model comparison (host, local-models env), then the report
PY=services/local-models/.venv/bin/python
$PY eval/compare_models.py encode --model ai-sage/Giga-Embeddings-instruct-480M-0826
$PY eval/compare_models.py report
python3 eval/report.py --runs balanced-9b
```

`eval/work/` holds intermediate artifacts (canonical texts, raw responses) and is
not committed.
