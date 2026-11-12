# Fresh-copy acceptance (stage 9)

Executed 2026-11-12 by the agent on macOS Apple Silicon (M4 Max), host Ollama 0.40.16.

| Step | Command | Result |
|---|---|---|
| Fresh copy | `git clone https://github.com/texpomru13/b1-test.git` (HEAD `d29a210`) into an empty scratch dir; no `.runtime`, no `.env` | done |
| First launch | `B1_PROJECT=b1-release ./run --quality balanced --port 8001` | exit 0, Ready, wall **20 s** |
| Resolved stack | balanced: Giga-Embeddings-instruct-480M + bge-reranker-v2-m3 + qwen3.5:9b-q4_K_M, `LLM_THINKING=off` from the profile | `resolved.env` |
| Demo (upload 2 PDFs, poll, list, exact, paraphrase, no-answer, debug, isolation, 2nd doc) | `sh eval/demo.sh http://127.0.0.1:8001` | exit 0, wall 362 s (incl. indexing of 2 docs); outcomes as expected; answers identical to the b1-eval stack |
| Stop | `./run stop` | exit 0, 0 containers left |
| Restart | same launch command | exit 0, wall **17 s**; `ollama pull` only verified the manifest (no layer re-download); models from the volume |
| Persistence | `GET /api/v1/documents` after restart | both documents `succeeded`, `ready=true` |
| First QA after restart | exact question on doc A | 200, same answer, 62 s (cold model load into Ollama under a concurrent judge job) |

Honest limits of this measurement:

- The 20 s first launch is **warm-cache**: Docker BuildKit layers were cached (same
  Dockerfiles built earlier for `b1-eval`), the HF models were seeded from the host
  cache (`B1_HF_SEED_DIR`), and the Ollama model was already pulled. A cold machine
  (image build from scratch + HF model download + 6.5 GB Ollama layers) was **not_run**.
- Timings were taken while the local judge (qwen3.6:27b) was running in parallel on
  the same host, so QA latency is pessimistic.
