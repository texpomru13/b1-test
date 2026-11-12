# Stage-9 evaluation report

Generated 2026-11-12T06:09:05+00:00 from artifacts in `eval/work/` (see `eval/README.md` for how to reproduce).

Gold set: 133 cases over 14 documents (dev 35, holdout 98); agent-drafted, quotes machine-verified, NOT human-reviewed.

## Summary (final, stage 9)

Executed on macOS Apple Silicon (M4 Max), host Ollama 0.40.16, stack `b1-eval`
(isolated compose project; the default `b1-rag` project and its volumes were not touched).
Profile **balanced**: Giga-Embeddings-instruct-480M + bge-reranker-v2-m3 + qwen3.5:9b-q4_K_M.
Judge: local qwen3.6:27b-q4_K_M (no paid API calls). Gold: 133 agent-drafted cases
(93 answerable, 40 no-answer) + 17 cross-document isolation probes; dev = documents
01/05/13/15, holdout = the other 10 documents.

**Final preset: `balanced-9b-t0`** = thinking off + quote matching v2 (PDF intra-word
spaces, ordered ellipsis gaps) + deterministic sampling (temperature 0, presence
penalty 0, seed 0). Chosen on DEV. Caveat: the holdout misses of earlier variants were
inspected while diagnosing the provenance failures, so the holdout numbers of the two
fixes are **post-hoc, not a blind estimate**.

| Variant | Split | Answerable found | Correct abstention | 5xx | Isolation | Fact recall (all answerable) | Complete answers | Forbidden facts |
|---|---|---|---|---|---|---|---|---|
| thinking auto | dev | 15/24 | 11/11 | 5/24 | — | — | — | — |
| thinking off | holdout | 46/69 | 27/29 | 0 | 14/14 | 90/181 = 50% | 30/69 | 0 |
| + quote match v2 | holdout | 54/69 | 27/29 | 0 | 14/14 | — | — | — |
| **+ temperature 0** | holdout | **60/69** | **28/29** | **0** | **14/14** | **128/181 = 71%** | **37/69** | **0** |
| **+ temperature 0** | all | **79/93** | **39/40** | **0** | **17/17** | **171/248 = 69%** | **45/93** | **0** |
| low-memory 4B, thinking off | holdout | 44/69 | 29/29 | 1 | 14/14 | 78/177 = 44% | 22/68 | 0 |

Paired sign test (correct outcome per gold case), `eval/reports/runs/balanced-9b-t0/paired.txt`:
t0 vs thinking-off — holdout 15 improved / 0 regressed (p=0.0001), all 20/3 (p=0.0005);
t0 vs quote-match-v2 — holdout 8/1 (p=0.039), dev 5/2 (p=0.45).

Invariants on every run: citation span validity 100%, cross-document leakage 0,
schema-valid 200 bodies 100% (one invalid body before commit a69d620, fixed),
retrieval hit@60 = 93/93.

False positives (found=true on a no-answer case), adjudicated by the judge from the
verified quotes: t0 — 1 (case 10-09, quotes do not answer → hallucination);
thinking-off — 3 (01-07, 08-07, 10-09). No isolation probe ever returned found=true.

Why answers were missed (thinking-off run, 29 misses, re-asked with log capture,
`miss_diagnosis.jsonl`): 18 — model quote not an exact substring of the canonical text
(provenance refusal); 9 — answered on re-ask (sampling noise); 2 — verifier rejected.
Captured rejected quotes: intra-word spaces in PDF text and ellipsis-joined fragments
→ fixed by quote matching v2; sampling noise → fixed by temperature 0. The 14
remaining t0 misses were not re-diagnosed (`not_run`).

**Latency** (200 responses): thinking off p50 19 s / p95 45 s (stack alone);
4B p50 15 s / p95 30 s. The t0 run's p50 49 s / p95 104 s was measured while the
27B judge was running on the same GPU, so it is NOT comparable; uncontended t0
latency was not re-measured (`not_run`).

**Retrieval** (gold spans, published index, all split): hybrid RRF + rerank
recall@8 0.968, all-spans@8 0.968, hit@60 1.0, nDCG@10 0.938. Dense alone recall@8
0.964; hybrid RRF without reranker 0.943; BM25 alone 0.863. The reranker is what
makes hybrid beat dense at top-8.
Encoders (dense recall@8, same chunks): Giga-3B 0.975, Giga-480M 0.964, BGE-M3 0.931;
480M encodes ~8× faster than 3B. Rerankers on frozen top-60 candidates (all-spans@8):
bge-reranker-v2-m3 0.968, Qwen3-Reranker-0.6B 0.935–0.946 (7× slower).
**Qwen3-Reranker-8B: `not_run`** — fp32 on MPS occupied ~100 GB of unified memory,
swap reached 39/40 GB and no case finished in 2 h 26 min; stopped to protect the host.

**RAGAS** (ragas 0.4.3, local judge), thinking-off run, all 93 answerable:
faithfulness 0.95, context recall 0.968, context precision 0.833, factual
correctness F1 0.635 (found answers). Context metrics depend only on retrieval and
are identical in nature for t0. RAGAS for t0 on holdout: see the t0 run section
(may be partial if still running at generation time).

Fresh-copy acceptance and demo: `eval/reports/acceptance/ACCEPTANCE.md`.

Limits: gold is agent-drafted and not human-reviewed; the judge is a local 27B model
(no human agreement study); one machine; Linux/NVIDIA/Windows and cloud providers `not_run`.

---

### Run `balanced-9b-t0`

Started 2026-11-11T23:39:52.381632+00:00, API http://127.0.0.1:8000, git `a2c35fa5e28a`.

**Split `all`** — requests 150 (gold 133: answerable 93, no-answer 40; isolation probes 17)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 136/136 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/136 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 93/93 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 90/93 = 96.8% | PASS |
| Correct abstention on negatives | >=95% | 39/40 = 97.5% | PASS |
| Structured response validity | 100% | 150/150 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 17/17 = 100.0% | — |
| Answerable → found=true | — | 79/93 = 85.0% | — |
| Answerable → found=false (missed) | — | 14/93 = 15.0% | — |
| Answerable → error (5xx/timeout) | 0 | 0/93 = 0.0% | — |
| No-answer → error | 0 | 0/40 = 0.0% | — |
| found=true citing a gold span | — | 79/79 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9677 | — |
| nDCG@10 (reranked) | — | 0.9378 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 26/30 = 86.7% |
| multi | 12/16 = 75.0% |
| negation | 17/18 = 94.4% |
| no_answer | 39/40 = 97.5% |
| paraphrase | 24/29 = 82.8% |

Latency of 200 responses (client wall time, sequential): n=150, p50=49.2s, p95=103.7s, max=174.8s. Stage p50 ms: {'context': 3, 'encode': 156, 'fusion': 5, 'rerank': 26493}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 171/248 = 69.0% |
| Expected-fact recall, found=true only | 171/208 = 82.2% |
| Fully complete answers (all facts) | 45/93 |
| Answers asserting a forbidden fact | 0/93 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 1/1 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/1 |

False positives (found=true where not-found was expected) — every case:

| Request | Judge: quotes answer? | Judge reason |
|---|---|---|
| 10-09 | False | Цитаты указывают, что размер расходов определяется соглашением сторон или структурой тарифной ставки, но не содержат конкретного процентного значения. |

**Split `holdout`** — requests 112 (gold 98: answerable 69, no-answer 29; isolation probes 14)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 107/107 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/107 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 69/69 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 67/69 = 97.1% | PASS |
| Correct abstention on negatives | >=95% | 28/29 = 96.5% | PASS |
| Structured response validity | 100% | 112/112 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 14/14 = 100.0% | — |
| Answerable → found=true | — | 60/69 = 87.0% | — |
| Answerable → found=false (missed) | — | 9/69 = 13.0% | — |
| Answerable → error (5xx/timeout) | 0 | 0/69 = 0.0% | — |
| No-answer → error | 0 | 0/29 = 0.0% | — |
| found=true citing a gold span | — | 60/60 = 100.0% | — |
| Mean gold-span recall in context | — | 0.971 | — |
| nDCG@10 (reranked) | — | 0.9306 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 19/22 = 86.4% |
| multi | 9/12 = 75.0% |
| negation | 13/14 = 92.9% |
| no_answer | 28/29 = 96.5% |
| paraphrase | 19/21 = 90.5% |

Latency of 200 responses (client wall time, sequential): n=112, p50=58.5s, p95=108.2s, max=174.8s. Stage p50 ms: {'context': 3, 'encode': 160, 'fusion': 6, 'rerank': 31888}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 128/181 = 70.7% |
| Expected-fact recall, found=true only | 128/153 = 83.7% |
| Fully complete answers (all facts) | 37/69 |
| Answers asserting a forbidden fact | 0/69 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 1/1 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/1 |

**Split `dev`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 29/29 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/29 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 24/24 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 23/24 = 95.8% | PASS |
| Correct abstention on negatives | >=95% | 11/11 = 100.0% | PASS |
| Structured response validity | 100% | 38/38 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 19/24 = 79.2% | — |
| Answerable → found=false (missed) | — | 5/24 = 20.8% | — |
| Answerable → error (5xx/timeout) | 0 | 0/24 = 0.0% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 19/19 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9583 | — |
| nDCG@10 (reranked) | — | 0.9584 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 7/8 = 87.5% |
| multi | 3/4 = 75.0% |
| negation | 4/4 = 100.0% |
| no_answer | 11/11 = 100.0% |
| paraphrase | 5/8 = 62.5% |

Latency of 200 responses (client wall time, sequential): n=38, p50=38.4s, p95=74.8s, max=103.7s. Stage p50 ms: {'context': 2, 'encode': 148, 'fusion': 5, 'rerank': 20454}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 43/67 = 64.2% |
| Expected-fact recall, found=true only | 43/55 = 78.2% |
| Fully complete answers (all facts) | 8/24 |
| Answers asserting a forbidden fact | 0/24 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 0/0 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/0 |

### Run `balanced-9b-nothink-qm2`

Started 2026-11-11T21:28:05.088901+00:00, API http://127.0.0.1:8000, git `a2c35fa5e28a`.

**Split `all`** — requests 150 (gold 133: answerable 93, no-answer 40; isolation probes 17)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 144/144 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/144 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 93/93 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 90/93 = 96.8% | PASS |
| Correct abstention on negatives | >=95% | 37/40 = 92.5% | FAIL |
| Structured response validity | 100% | 150/150 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 17/17 = 100.0% | — |
| Answerable → found=true | — | 71/93 = 76.3% | — |
| Answerable → found=false (missed) | — | 22/93 = 23.7% | — |
| Answerable → error (5xx/timeout) | 0 | 0/93 = 0.0% | — |
| No-answer → error | 0 | 0/40 = 0.0% | — |
| found=true citing a gold span | — | 71/71 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9677 | — |
| nDCG@10 (reranked) | — | 0.9378 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 24/30 = 80.0% |
| multi | 11/16 = 68.8% |
| negation | 12/18 = 66.7% |
| no_answer | 37/40 = 92.5% |
| paraphrase | 24/29 = 82.8% |

Latency of 200 responses (client wall time, sequential): n=150, p50=47.3s, p95=111.5s, max=154.1s. Stage p50 ms: {'context': 2, 'encode': 159, 'fusion': 5, 'rerank': 24019}.

**Split `holdout`** — requests 112 (gold 98: answerable 69, no-answer 29; isolation probes 14)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 110/110 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/110 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 69/69 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 67/69 = 97.1% | PASS |
| Correct abstention on negatives | >=95% | 27/29 = 93.1% | FAIL |
| Structured response validity | 100% | 112/112 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 14/14 = 100.0% | — |
| Answerable → found=true | — | 54/69 = 78.3% | — |
| Answerable → found=false (missed) | — | 15/69 = 21.7% | — |
| Answerable → error (5xx/timeout) | 0 | 0/69 = 0.0% | — |
| No-answer → error | 0 | 0/29 = 0.0% | — |
| found=true citing a gold span | — | 54/54 = 100.0% | — |
| Mean gold-span recall in context | — | 0.971 | — |
| nDCG@10 (reranked) | — | 0.9306 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 18/22 = 81.8% |
| multi | 7/12 = 58.3% |
| negation | 10/14 = 71.4% |
| no_answer | 27/29 = 93.1% |
| paraphrase | 19/21 = 90.5% |

Latency of 200 responses (client wall time, sequential): n=112, p50=49.4s, p95=114.1s, max=154.1s. Stage p50 ms: {'context': 3, 'encode': 158, 'fusion': 5, 'rerank': 26777}.

**Split `dev`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 34/34 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/34 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 24/24 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 23/24 = 95.8% | PASS |
| Correct abstention on negatives | >=95% | 10/11 = 90.9% | FAIL |
| Structured response validity | 100% | 38/38 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 17/24 = 70.8% | — |
| Answerable → found=false (missed) | — | 7/24 = 29.2% | — |
| Answerable → error (5xx/timeout) | 0 | 0/24 = 0.0% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 17/17 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9583 | — |
| nDCG@10 (reranked) | — | 0.9584 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 6/8 = 75.0% |
| multi | 4/4 = 100.0% |
| negation | 2/4 = 50.0% |
| no_answer | 10/11 = 90.9% |
| paraphrase | 5/8 = 62.5% |

Latency of 200 responses (client wall time, sequential): n=38, p50=38.9s, p95=103.9s, max=127.1s. Stage p50 ms: {'context': 2, 'encode': 160, 'fusion': 5, 'rerank': 21254}.

### Run `balanced-9b-nothink`

Started 2026-11-11T12:29:59.548112+00:00, API http://127.0.0.1:8000, git `29bd1048b7f3`.

**Split `all`** — requests 150 (gold 133: answerable 93, no-answer 40; isolation probes 17)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 124/124 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/124 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 93/93 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 90/93 = 96.8% | PASS |
| Correct abstention on negatives | >=95% | 37/40 = 92.5% | FAIL |
| Structured response validity | 100% | 149/150 = 99.3% | FAIL |
| Isolation probes answered found=false | 100% | 17/17 = 100.0% | — |
| Answerable → found=true | — | 64/93 = 68.8% | — |
| Answerable → found=false (missed) | — | 29/93 = 31.2% | — |
| Answerable → error (5xx/timeout) | 0 | 0/93 = 0.0% | — |
| No-answer → error | 0 | 0/40 = 0.0% | — |
| found=true citing a gold span | — | 64/64 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9677 | — |
| nDCG@10 (reranked) | — | 0.9378 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 23/30 = 76.7% |
| multi | 7/16 = 43.8% |
| negation | 10/18 = 55.6% |
| no_answer | 37/40 = 92.5% |
| paraphrase | 24/29 = 82.8% |

Latency of 200 responses (client wall time, sequential): n=150, p50=19.1s, p95=45.2s, max=69.3s. Stage p50 ms: {'context': 1, 'encode': 142, 'fusion': 3, 'rerank': 9347}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 132/248 = 53.2% |
| Expected-fact recall, found=true only | 132/158 = 83.5% |
| Fully complete answers (all facts) | 41/93 |
| Answers asserting a forbidden fact | 0/93 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 3/3 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/3 |
| RAGAS faithfulness[found] | 0.950 (n=64, NaN=0) |
| RAGAS faithfulness[all] | 0.950 (n=64, NaN=0) |
| RAGAS context_recall[found] | 1.000 (n=64, NaN=0) |
| RAGAS context_recall[all] | 0.968 (n=93, NaN=0) |
| RAGAS llm_context_precision_with_reference[found] | 0.924 (n=64, NaN=0) |
| RAGAS llm_context_precision_with_reference[all] | 0.833 (n=93, NaN=0) |
| RAGAS factual_correctness(mode=f1)[found] | 0.635 (n=64, NaN=0) |
| RAGAS factual_correctness(mode=f1)[all] | 0.635 (n=64, NaN=0) |

False positives (found=true where not-found was expected) — every case:

| Request | Judge: quotes answer? | Judge reason |
|---|---|---|
| 01-07 | False | Цитаты содержат только пропуски (подчеркивания), не раскрывая фактическое наименование организации и ФИО подписанта, поэтому вопрос по существу не решен. |
| 08-07 | False | В цитатах вместо конкретного числового значения тарифа присутствуют пропуски (_______), поэтому вопрос о размере тарифа в рублях не может быть решен. |
| 10-09 | False | Цитаты указывают, что размер расходов определяется соглашением сторон или структурой тарифной ставки, но не содержат конкретного процентного значения. |

**Split `holdout`** — requests 112 (gold 98: answerable 69, no-answer 29; isolation probes 14)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 85/85 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/85 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 69/69 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 67/69 = 97.1% | PASS |
| Correct abstention on negatives | >=95% | 27/29 = 93.1% | FAIL |
| Structured response validity | 100% | 112/112 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 14/14 = 100.0% | — |
| Answerable → found=true | — | 46/69 = 66.7% | — |
| Answerable → found=false (missed) | — | 23/69 = 33.3% | — |
| Answerable → error (5xx/timeout) | 0 | 0/69 = 0.0% | — |
| No-answer → error | 0 | 0/29 = 0.0% | — |
| found=true citing a gold span | — | 46/46 = 100.0% | — |
| Mean gold-span recall in context | — | 0.971 | — |
| nDCG@10 (reranked) | — | 0.9306 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 16/22 = 72.7% |
| multi | 5/12 = 41.7% |
| negation | 8/14 = 57.1% |
| no_answer | 27/29 = 93.1% |
| paraphrase | 17/21 = 81.0% |

Latency of 200 responses (client wall time, sequential): n=112, p50=21.4s, p95=48.2s, max=69.3s. Stage p50 ms: {'context': 1, 'encode': 142, 'fusion': 4, 'rerank': 9334}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 90/181 = 49.7% |
| Expected-fact recall, found=true only | 90/109 = 82.6% |
| Fully complete answers (all facts) | 30/69 |
| Answers asserting a forbidden fact | 0/69 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 2/2 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/2 |
| RAGAS faithfulness[found] | 0.967 (n=46, NaN=0) |
| RAGAS faithfulness[all] | 0.967 (n=46, NaN=0) |
| RAGAS context_recall[found] | 1.000 (n=46, NaN=0) |
| RAGAS context_recall[all] | 0.971 (n=69, NaN=0) |
| RAGAS llm_context_precision_with_reference[found] | 0.894 (n=46, NaN=0) |
| RAGAS llm_context_precision_with_reference[all] | 0.811 (n=69, NaN=0) |
| RAGAS factual_correctness(mode=f1)[found] | 0.617 (n=46, NaN=0) |
| RAGAS factual_correctness(mode=f1)[all] | 0.617 (n=46, NaN=0) |

**Split `dev`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 39/39 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/39 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 24/24 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 23/24 = 95.8% | PASS |
| Correct abstention on negatives | >=95% | 10/11 = 90.9% | FAIL |
| Structured response validity | 100% | 37/38 = 97.4% | FAIL |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 18/24 = 75.0% | — |
| Answerable → found=false (missed) | — | 6/24 = 25.0% | — |
| Answerable → error (5xx/timeout) | 0 | 0/24 = 0.0% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 18/18 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9583 | — |
| nDCG@10 (reranked) | — | 0.9584 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 7/8 = 87.5% |
| multi | 2/4 = 50.0% |
| negation | 2/4 = 50.0% |
| no_answer | 10/11 = 90.9% |
| paraphrase | 7/8 = 87.5% |

Latency of 200 responses (client wall time, sequential): n=38, p50=16.9s, p95=42.6s, max=45.2s. Stage p50 ms: {'context': 1, 'encode': 142, 'fusion': 3, 'rerank': 9561}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 42/67 = 62.7% |
| Expected-fact recall, found=true only | 42/49 = 85.7% |
| Fully complete answers (all facts) | 11/24 |
| Answers asserting a forbidden fact | 0/24 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 1/1 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/1 |
| RAGAS faithfulness[found] | 0.906 (n=18, NaN=0) |
| RAGAS faithfulness[all] | 0.906 (n=18, NaN=0) |
| RAGAS context_recall[found] | 1.000 (n=18, NaN=0) |
| RAGAS context_recall[all] | 0.958 (n=24, NaN=0) |
| RAGAS llm_context_precision_with_reference[found] | 1.000 (n=18, NaN=0) |
| RAGAS llm_context_precision_with_reference[all] | 0.894 (n=24, NaN=0) |
| RAGAS factual_correctness(mode=f1)[found] | 0.679 (n=18, NaN=0) |
| RAGAS factual_correctness(mode=f1)[all] | 0.679 (n=18, NaN=0) |

### Run `balanced-9b`

Started 2026-11-11T12:17:18.279870+00:00, API http://127.0.0.1:8000, git `29bd1048b7f3`.

**Split `all`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 24/24 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/24 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 19/24 = 79.2% | FAIL |
| Context evidence coverage (all spans) | >=90% | 18/24 = 75.0% | FAIL |
| Correct abstention on negatives | >=95% | 11/11 = 100.0% | PASS |
| Structured response validity | 100% | 33/33 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 15/24 = 62.5% | — |
| Answerable → found=false (missed) | — | 4/24 = 16.7% | — |
| Answerable → error (5xx/timeout) | 0 | 5/24 = 20.8% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 15/15 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9474 | — |
| nDCG@10 (reranked) | — | 0.9626 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 4/8 = 50.0% |
| multi | 2/4 = 50.0% |
| negation | 3/4 = 75.0% |
| no_answer | 11/11 = 100.0% |
| paraphrase | 6/8 = 75.0% |

Latency of 200 responses (client wall time, sequential): n=33, p50=144.9s, p95=382.0s, max=534.5s. Stage p50 ms: {'context': 1, 'encode': 320, 'fusion': 6, 'rerank': 9571}.

**Split `holdout`** — requests 0 (gold 0: answerable 0, no-answer 0; isolation probes 0)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | n/a (0 cases) | not_run |
| Cross-document leakage (citations) | 0 | n/a (0 cases) | not_run |
| Positive retrieval hit@candidates (60) | >=95% | n/a (0 cases) | not_run |
| Context evidence coverage (all spans) | >=90% | n/a (0 cases) | not_run |
| Correct abstention on negatives | >=95% | n/a (0 cases) | not_run |
| Structured response validity | 100% | n/a (0 cases) | not_run |
| Isolation probes answered found=false | 100% | n/a (0 cases) | — |
| Answerable → found=true | — | n/a (0 cases) | — |
| Answerable → found=false (missed) | — | n/a (0 cases) | — |
| Answerable → error (5xx/timeout) | 0 | n/a (0 cases) | — |
| No-answer → error | 0 | n/a (0 cases) | — |
| found=true citing a gold span | — | n/a (0 cases) | — |
| Mean gold-span recall in context | — | None | — |
| nDCG@10 (reranked) | — | None | — |

**Split `dev`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 24/24 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/24 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 19/24 = 79.2% | FAIL |
| Context evidence coverage (all spans) | >=90% | 18/24 = 75.0% | FAIL |
| Correct abstention on negatives | >=95% | 11/11 = 100.0% | PASS |
| Structured response validity | 100% | 33/33 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 15/24 = 62.5% | — |
| Answerable → found=false (missed) | — | 4/24 = 16.7% | — |
| Answerable → error (5xx/timeout) | 0 | 5/24 = 20.8% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 15/15 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9474 | — |
| nDCG@10 (reranked) | — | 0.9626 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 4/8 = 50.0% |
| multi | 2/4 = 50.0% |
| negation | 3/4 = 75.0% |
| no_answer | 11/11 = 100.0% |
| paraphrase | 6/8 = 75.0% |

Latency of 200 responses (client wall time, sequential): n=33, p50=144.9s, p95=382.0s, max=534.5s. Stage p50 ms: {'context': 1, 'encode': 320, 'fusion': 6, 'rerank': 9571}.

### Run `low-4b-off`

Started 2026-11-11T17:58:27.466780+00:00, API http://127.0.0.1:8000, git `03896f8489e4`.

**Split `all`** — requests 150 (gold 133: answerable 93, no-answer 40; isolation probes 17)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 103/103 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/103 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 92/93 = 98.9% | PASS |
| Context evidence coverage (all spans) | >=90% | 89/93 = 95.7% | PASS |
| Correct abstention on negatives | >=95% | 40/40 = 100.0% | PASS |
| Structured response validity | 100% | 149/149 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 17/17 = 100.0% | — |
| Answerable → found=true | — | 59/93 = 63.4% | — |
| Answerable → found=false (missed) | — | 33/93 = 35.5% | — |
| Answerable → error (5xx/timeout) | 0 | 1/93 = 1.1% | — |
| No-answer → error | 0 | 0/40 = 0.0% | — |
| found=true citing a gold span | — | 58/59 = 98.3% | — |
| Mean gold-span recall in context | — | 0.9674 | — |
| nDCG@10 (reranked) | — | 0.9371 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 20/30 = 66.7% |
| multi | 7/16 = 43.8% |
| negation | 10/18 = 55.6% |
| no_answer | 40/40 = 100.0% |
| paraphrase | 22/29 = 75.9% |

Latency of 200 responses (client wall time, sequential): n=149, p50=14.7s, p95=30.3s, max=55.1s. Stage p50 ms: {'context': 1, 'encode': 136, 'fusion': 3, 'rerank': 9212}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 110/244 = 45.1% |
| Expected-fact recall, found=true only | 110/145 = 75.9% |
| Fully complete answers (all facts) | 32/92 |
| Answers asserting a forbidden fact | 0/92 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 0/0 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/0 |

**Split `holdout`** — requests 112 (gold 98: answerable 69, no-answer 29; isolation probes 14)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 77/77 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/77 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 68/69 = 98.6% | PASS |
| Context evidence coverage (all spans) | >=90% | 66/69 = 95.7% | PASS |
| Correct abstention on negatives | >=95% | 29/29 = 100.0% | PASS |
| Structured response validity | 100% | 111/111 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 14/14 = 100.0% | — |
| Answerable → found=true | — | 44/69 = 63.8% | — |
| Answerable → found=false (missed) | — | 24/69 = 34.8% | — |
| Answerable → error (5xx/timeout) | 0 | 1/69 = 1.5% | — |
| No-answer → error | 0 | 0/29 = 0.0% | — |
| found=true citing a gold span | — | 44/44 = 100.0% | — |
| Mean gold-span recall in context | — | 0.9706 | — |
| nDCG@10 (reranked) | — | 0.9296 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 14/22 = 63.6% |
| multi | 6/12 = 50.0% |
| negation | 8/14 = 57.1% |
| no_answer | 29/29 = 100.0% |
| paraphrase | 16/21 = 76.2% |

Latency of 200 responses (client wall time, sequential): n=111, p50=15.4s, p95=31.4s, max=55.1s. Stage p50 ms: {'context': 1, 'encode': 139, 'fusion': 3, 'rerank': 9646}.

LLM-judged (judge `qwen3.6:27b-q4_K_M`, local):

| Metric | Value |
|---|---|
| Expected-fact recall, all answerable | 78/177 = 44.1% |
| Expected-fact recall, found=true only | 78/106 = 73.6% |
| Fully complete answers (all facts) | 22/68 |
| Answers asserting a forbidden fact | 0/68 |
| Judge output errors | 0 |
| found=true on no-answer/isolation: judged hallucination | 0/0 |
| found=true on no-answer/isolation: quotes do answer (gold error?) | 0/0 |

**Split `dev`** — requests 38 (gold 35: answerable 24, no-answer 11; isolation probes 3)

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| Citation span validity | 100% | 26/26 = 100.0% | PASS |
| Cross-document leakage (citations) | 0 | 0/26 = 0.0% | PASS |
| Positive retrieval hit@candidates (60) | >=95% | 24/24 = 100.0% | PASS |
| Context evidence coverage (all spans) | >=90% | 23/24 = 95.8% | PASS |
| Correct abstention on negatives | >=95% | 11/11 = 100.0% | PASS |
| Structured response validity | 100% | 38/38 = 100.0% | PASS |
| Isolation probes answered found=false | 100% | 3/3 = 100.0% | — |
| Answerable → found=true | — | 15/24 = 62.5% | — |
| Answerable → found=false (missed) | — | 9/24 = 37.5% | — |
| Answerable → error (5xx/timeout) | 0 | 0/24 = 0.0% | — |
| No-answer → error | 0 | 0/11 = 0.0% | — |
| found=true citing a gold span | — | 14/15 = 93.3% | — |
| Mean gold-span recall in context | — | 0.9583 | — |
| nDCG@10 (reranked) | — | 0.9584 | — |

Per category (answerable: found=true; no_answer: found=false):

| Category | Correct outcome |
|---|---|
| exact | 6/8 = 75.0% |
| multi | 1/4 = 25.0% |
| negation | 2/4 = 50.0% |
| no_answer | 11/11 = 100.0% |
| paraphrase | 6/8 = 75.0% |

Latency of 200 responses (client wall time, sequential): n=38, p50=12.4s, p95=27.3s, max=39.0s. Stage p50 ms: {'context': 1, 'encode': 130, 'fusion': 3, 'rerank': 8871}.

## Retrieval ablation (same gold spans, published balanced index)

Split `all` (n=93 answerable):

| Method | recall@1 | recall@5 | recall@8 | recall@20 | recall@60 | hit@60 | all_spans@8 | ndcg@10 | mrr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.7943 | 0.9487 | 0.9642 | 0.9785 | 1.0 | 1.0 | 0.9462 | 0.9163 | 0.9194 |
| sparse | 0.5581 | 0.8054 | 0.8631 | 0.9211 | 0.9695 | 0.9785 | 0.828 | 0.7426 | 0.7326 |
| hybrid_rrf | 0.6996 | 0.9201 | 0.9427 | 0.9785 | 1.0 | 1.0 | 0.9247 | 0.8675 | 0.8595 |
| hybrid_dbsf | 0.7817 | 0.9004 | 0.9337 | 0.9749 | 1.0 | 1.0 | 0.9032 | 0.8936 | 0.9032 |
| rrf_rerank | 0.848 | 0.9434 | 0.9677 | 1.0 | 1.0 | 1.0 | 0.9677 | 0.9378 | 0.9398 |

Split `holdout` (n=69 answerable):

| Method | recall@1 | recall@5 | recall@8 | recall@20 | recall@60 | hit@60 | all_spans@8 | ndcg@10 | mrr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.7662 | 0.9357 | 0.9565 | 0.971 | 1.0 | 1.0 | 0.942 | 0.8994 | 0.9059 |
| sparse | 0.5275 | 0.7957 | 0.8589 | 0.8937 | 0.9589 | 0.971 | 0.8116 | 0.7209 | 0.7205 |
| hybrid_rrf | 0.6821 | 0.8971 | 0.9275 | 0.971 | 1.0 | 1.0 | 0.913 | 0.8558 | 0.8541 |
| hybrid_dbsf | 0.7638 | 0.8899 | 0.93 | 0.9662 | 1.0 | 1.0 | 0.8986 | 0.8815 | 0.9005 |
| rrf_rerank | 0.8386 | 0.9382 | 0.971 | 1.0 | 1.0 | 1.0 | 0.971 | 0.9306 | 0.9391 |

## Encoder / reranker comparison (fixed chunks, frozen candidates)

| Encoder | Mode | recall@1 | recall@5 | recall@8 | recall@20 | recall@60 | all_spans@8 | ndcg@10 | encode s |
|---|---|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | dense[all] | 0.7228 | 0.9129 | 0.9308 | 0.9606 | 1.0 | 0.914 | 0.8639 | 33.2 |
| BAAI/bge-m3 | hybrid_rrf60[all] | 0.6828 | 0.8771 | 0.9247 | 0.9516 | 1.0 | 0.8925 | 0.8289 | 33.2 |
| BAAI/bge-m3 | dense[holdout] | 0.6843 | 0.9116 | 0.9309 | 0.9614 | 1.0 | 0.913 | 0.85 | 33.2 |
| BAAI/bge-m3 | hybrid_rrf60[holdout] | 0.6594 | 0.8778 | 0.9227 | 0.9493 | 1.0 | 0.8841 | 0.8209 | 33.2 |
| ai-sage/Giga-Embeddings-instruct-480M-0826 | dense[all] | 0.7943 | 0.9487 | 0.9642 | 0.9785 | 1.0 | 0.9462 | 0.9163 | 39.7 |
| ai-sage/Giga-Embeddings-instruct-480M-0826 | hybrid_rrf60[all] | 0.7194 | 0.9004 | 0.9283 | 0.9695 | 1.0 | 0.8925 | 0.862 | 39.7 |
| ai-sage/Giga-Embeddings-instruct-480M-0826 | dense[holdout] | 0.7662 | 0.9357 | 0.9565 | 0.971 | 1.0 | 0.942 | 0.8994 | 39.7 |
| ai-sage/Giga-Embeddings-instruct-480M-0826 | hybrid_rrf60[holdout] | 0.7087 | 0.8899 | 0.9082 | 0.9589 | 1.0 | 0.8696 | 0.8524 | 39.7 |
| ai-sage/Giga-Embeddings-instruct | dense[all] | 0.8427 | 0.9466 | 0.9749 | 0.9892 | 1.0 | 0.9677 | 0.9353 | 315.9 |
| ai-sage/Giga-Embeddings-instruct | hybrid_rrf60[all] | 0.7355 | 0.895 | 0.9355 | 0.9803 | 1.0 | 0.914 | 0.8696 | 315.9 |
| ai-sage/Giga-Embeddings-instruct | dense[holdout] | 0.8314 | 0.9329 | 0.9662 | 0.9855 | 1.0 | 0.9565 | 0.9255 | 315.9 |
| ai-sage/Giga-Embeddings-instruct | hybrid_rrf60[holdout] | 0.7304 | 0.8826 | 0.9275 | 0.9734 | 1.0 | 0.8986 | 0.8592 | 315.9 |

| Reranker | Candidates from | recall@1 | recall@5 | recall@8 | recall@20 | recall@60 | all_spans@8 | ndcg@10 | score s |
|---|---|---|---|---|---|---|---|---|---|
| BAAI/bge-reranker-v2-m3 | ai-sage/Giga-Embeddings-instruct[all] | 0.848 | 0.9434 | 0.9677 | 1.0 | 1.0 | 0.9677 | 0.9378 | 122.2 |
| BAAI/bge-reranker-v2-m3 | ai-sage/Giga-Embeddings-instruct[holdout] | 0.8386 | 0.9382 | 0.971 | 1.0 | 1.0 | 0.971 | 0.9306 | 122.2 |
| BAAI/bge-reranker-v2-m3 | ai-sage/Giga-Embeddings-instruct-480M-0826[all] | 0.848 | 0.9434 | 0.9677 | 1.0 | 1.0 | 0.9677 | 0.9378 | 122.2 |
| BAAI/bge-reranker-v2-m3 | ai-sage/Giga-Embeddings-instruct-480M-0826[holdout] | 0.8386 | 0.9382 | 0.971 | 1.0 | 1.0 | 0.971 | 0.9306 | 122.2 |
| Qwen/Qwen3-Reranker-0.6B | ai-sage/Giga-Embeddings-instruct[all] | 0.8312 | 0.9373 | 0.9624 | 1.0 | 1.0 | 0.9462 | 0.9371 | 880.9 |
| Qwen/Qwen3-Reranker-0.6B | ai-sage/Giga-Embeddings-instruct[holdout] | 0.8111 | 0.93 | 0.9638 | 1.0 | 1.0 | 0.942 | 0.9263 | 880.9 |
| Qwen/Qwen3-Reranker-0.6B | ai-sage/Giga-Embeddings-instruct-480M-0826[all] | 0.8312 | 0.9373 | 0.9516 | 1.0 | 1.0 | 0.9355 | 0.937 | 880.9 |
| Qwen/Qwen3-Reranker-0.6B | ai-sage/Giga-Embeddings-instruct-480M-0826[holdout] | 0.8111 | 0.93 | 0.9493 | 1.0 | 1.0 | 0.9275 | 0.9262 | 880.9 |

