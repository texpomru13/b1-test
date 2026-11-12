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
