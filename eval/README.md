# Evaluation (stage 9)

Reproducible evaluation of the RAG service against a gold set built from
`test_documents/`. The archived report is `eval/reports/REPORT.md`; raw measured
artifacts of the reported runs are in `eval/reports/runs/`.

The published runs used a local LLM and a local Ollama judge (`qwen3.6:27b-q4_K_M`, thinking off, temperature 0).

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
| `report.py` | Builds `work/REPORT.md` from the current artifacts |
| `demo.sh` | API demo (upload, poll, list, exact, paraphrase, no-answer, debug, isolation) with saved outputs |

## Gold set

133 cases over 14 text documents (document 06 is a scan without a text layer and
is used only as the `OCR_REQUIRED` ingestion case). Categories: exact 30,
paraphrase 29, multi-condition 16, negation/limits 18, no-answer 40. The split is
by document: dev = fixtures 01, 05, 13, 15 (35 cases), holdout = the other 10
(98 cases). **Provenance** — drafted by LLM agents from the canonical text; every
quote is machine-verified as an exact substring; absence of no-answer facts was
checked by text search (recorded in `notes`). The set has NOT been reviewed by a
human lawyer — treat it as a regression corpus.

## Reproduce

```bash
# 1. an isolated stack (own volumes; does not touch the default b1-rag project)
export B1_EVAL_WORK_DIR="$PWD/eval/work/verification"
B1_PROJECT=b1-eval ./run --quality balanced
# 2. corpus, index export, gold
uv run python eval/upload_corpus.py
python3 eval/export_index.py --project b1-eval
python3 eval/build_gold.py
# 3. retrieval ablation and the QA run
python3 eval/ablation.py --project b1-eval --name balanced
uv run python eval/run_qa.py --run balanced-9b
for split in all dev holdout; do
  uv run python eval/metrics.py --run balanced-9b --split "$split"
done
# 4. LLM judge (separate env: cd eval && uv sync)
uv run --frozen --project eval python eval/judge.py --run balanced-9b --split all
# 5. model comparison (host, local-models env), then the report
uv sync --frozen --project services/local-models
PY=services/local-models/.venv/bin/python
$PY eval/compare_models.py encode --device cpu --model ai-sage/Giga-Embeddings-instruct-480M-0826
$PY eval/compare_models.py report
python3 eval/report.py --runs balanced-9b
```

`eval/work/` holds intermediate artifacts (canonical texts, raw responses) and is
not committed. Use a separate work directory and a new run name for each configuration.
The uploaded fixture IDs in corpus.json determine exactly which documents are exported.
Building gold writes the verified coordinates to `$B1_EVAL_WORK_DIR/gold.jsonl`.
The evaluation uses that file when present, with the committed gold set as a fallback.
If upload polling is interrupted, repeat the same command to resume saved jobs.
HTTP, transport and schema errors count as unsuccessful requests in deterministic metrics.
Judge fact and RAGAS scores cover valid HTTP 200 answers. The judge summary reports
coverage and failed responses. A limited judge run writes a separate summary.
Use a new run name to change the judge or the service configuration.
The current report is written to the work directory, or the path passed to `report.py --output`.

The HTTP evaluation uses the provider configured on the running service.
The example selects the local profile. The demo requires curl and jq and exits with
an error if an expected result or the status of an indexing job is wrong.
