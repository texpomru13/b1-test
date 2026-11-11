"""Offline encoder / reranker comparison on FIXED chunk boundaries and frozen candidates.

Runs on the host with the local-models environment (pinned transformers/ST and the
production adapters), ONE model per process so weights are released between runs:

    PY=services/local-models/.venv/bin/python
    $PY eval/compare_models.py encode --model ai-sage/Giga-Embeddings-instruct-480M-0826
    $PY eval/compare_models.py encode --model ai-sage/Giga-Embeddings-instruct
    $PY eval/compare_models.py encode --model BAAI/bge-m3
    $PY eval/compare_models.py rerank --model BAAI/bge-reranker-v2-m3 --encoders A,B
    $PY eval/compare_models.py report

Encoders: every chunk of the published balanced run (same boundaries for all) and
every answerable gold question are embedded; dense retrieval = exact cosine
within the document. "hybrid" fuses each encoder's dense ranking with the SAME
BM25 ranking from the ablation run via RRF (k=60) computed here — an offline
approximation used only to compare encoders under identical conditions; the
production path always fuses server-side in Qdrant.
Rerankers: the frozen top-60 hybrid candidates of the chosen encoders are scored
by each reranker with the production adapter (same instruction / recipe).
Outputs eval/work/compare/*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eval" / "work"
OUT = WORK / "compare"
sys.path.insert(0, str(ROOT / "services" / "local-models"))
sys.path.insert(0, str(ROOT / "eval"))

from spans import ndcg, span_covered  # noqa: E402

REVISIONS = {
    "ai-sage/Giga-Embeddings-instruct-480M-0826": "1763d603adac8057bd6482a708001b0ed2e0a903",
    "ai-sage/Giga-Embeddings-instruct": "2cf0fdc97194aaedf10ac0e6bf798834acd31042",
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    "Qwen/Qwen3-Reranker-8B": "77d193c791ed757ca307ee72715aa132723da912",
}
GIGA_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)
RRF_K = 60


def slug(model: str) -> str:
    return model.replace("/", "__")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def load_inputs() -> tuple[list[dict], dict[str, dict]]:
    gold = [c for c in load_jsonl(ROOT / "eval" / "gold" / "gold.jsonl") if c["answerable"]]
    index = {p.stem: json.loads(p.read_text(encoding="utf-8"))
             for p in (WORK / "index").glob("*.json")}
    return gold, index


def cmd_encode(model_id: str, device: str, batch: int) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    gold, index = load_inputs()
    t0 = time.monotonic()
    model = SentenceTransformer(model_id, revision=REVISIONS[model_id],
                                trust_remote_code=model_id.startswith("ai-sage/"),
                                device=device)
    load_s = time.monotonic() - t0
    is_giga = model_id.startswith("ai-sage/")
    ranked: dict[str, list[str]] = {}
    t1 = time.monotonic()
    n_chunks = 0
    for fid, doc in sorted(index.items()):
        cases = [c for c in gold if c["doc_fixture_id"] == fid]
        if not cases:
            continue
        ids = [c["chunk_id"] for c in doc["chunks"]]
        dv = model.encode([c["text"] for c in doc["chunks"]], batch_size=batch,
                          normalize_embeddings=True, convert_to_numpy=True)
        n_chunks += len(ids)
        qs = [(GIGA_QUERY_INSTRUCTION if is_giga else "") + c["question"] for c in cases]
        qv = model.encode(qs, batch_size=batch, normalize_embeddings=True,
                          convert_to_numpy=True)
        sims = qv @ dv.T
        for case, row in zip(cases, sims, strict=True):
            order = np.argsort(-row, kind="stable")[:60]
            ranked[case["case_id"]] = [ids[i] for i in order]
    encode_s = time.monotonic() - t1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"enc-{slug(model_id)}.json").write_text(json.dumps({
        "model": model_id, "revision": REVISIONS[model_id], "device": device,
        "load_s": round(load_s, 1), "encode_s": round(encode_s, 1), "n_chunks": n_chunks,
        "dense": ranked}))
    print(f"{model_id}: load {load_s:.0f}s encode {encode_s:.0f}s chunks={n_chunks}")


def rrf(*rankings: list[str], k: int = RRF_K, limit: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda c: (-score[c], c))[:limit]


def bm25_ranking() -> dict[str, list[str]]:
    raw = json.loads((WORK / "ablation" / "balanced.raw.json").read_text(encoding="utf-8"))
    return {cid: r["sparse"] for cid, r in raw.items()}


def metrics(ranked: dict[str, list[str]], gold: list[dict], index: dict[str, dict]) -> dict:
    rows = []
    for case in gold:
        if case["case_id"] not in ranked:
            continue
        rng = {c["chunk_id"]: (c["char_start"], c["char_end"])
               for c in index[case["doc_fixture_id"]]["chunks"]}
        r = [rng[c] for c in ranked[case["case_id"]] if c in rng]
        spans = case["gold_spans"]
        row = {f"recall@{k}": sum(span_covered(s, r[:k]) for s in spans) / len(spans)
               for k in (1, 5, 8, 20, 60)}
        row["all_spans@8"] = float(all(span_covered(s, r[:8]) for s in spans))
        row["ndcg@10"] = ndcg(r, spans, 10)
        rows.append(row)
    return {"n": len(rows), **{k: round(statistics.fmean(x[k] for x in rows), 4)
                               for k in rows[0]}}


def cmd_rerank(model_id: str, encoders: list[str], device: str, batch: int) -> None:
    import torch
    from app_lm.rerankers import CrossEncoderReranker, QwenYesNoReranker

    gold, index = load_inputs()
    bm25 = bm25_ranking()
    text = {c["chunk_id"]: c["text"] for d in index.values() for c in d["chunks"]}
    t0 = time.monotonic()
    if model_id.startswith("Qwen/"):
        rr = QwenYesNoReranker(model_id, REVISIONS[model_id], batch_size=batch)
        rr._model.to(device)  # same fp32 weights as production, on the host GPU
        orig = rr._score_batch

        def score_batch(query: str, docs: list[str]) -> list[float]:
            tok = rr._tokenizer
            real_pad = tok.pad

            def pad_to_device(*a, **kw):  # noqa: ANN002, ANN003
                return real_pad(*a, **kw).to(device)

            tok.pad = pad_to_device  # type: ignore[method-assign]
            try:
                return orig(query, docs)
            finally:
                tok.pad = real_pad  # type: ignore[method-assign]

        rr._score_batch = score_batch  # type: ignore[method-assign]
    else:
        rr = CrossEncoderReranker(model_id, REVISIONS[model_id], batch_size=batch)
        rr._model.model.to(device)
    load_s = time.monotonic() - t0
    cache: dict[tuple[str, str], float] = {}
    out: dict[str, dict[str, list[str]]] = {}
    t1 = time.monotonic()
    pairs = 0
    for enc in encoders:
        dense = json.loads((OUT / f"enc-{slug(enc)}.json").read_text())["dense"]
        out[enc] = {}
        for case in gold:
            cands = rrf(dense[case["case_id"]], bm25[case["case_id"]])
            need = [c for c in cands if (case["case_id"], c) not in cache]
            if need:
                with torch.no_grad():
                    scores = rr.score(case["question"], [text[c] for c in need])
                pairs += len(need)
                for c, s in zip(need, scores, strict=True):
                    cache[(case["case_id"], c)] = s
            out[enc][case["case_id"]] = sorted(
                cands, key=lambda c: (-cache[(case["case_id"], c)], c))
        print(f"  {enc} done, pairs so far {pairs}", flush=True)
    (OUT / f"rr-{slug(model_id)}.json").write_text(json.dumps({
        "model": model_id, "revision": REVISIONS[model_id], "device": device,
        "load_s": round(load_s, 1), "score_s": round(time.monotonic() - t1, 1),
        "pairs": pairs, "ranked": out}))
    print(f"{model_id}: load {load_s:.0f}s score {time.monotonic() - t1:.0f}s pairs={pairs}")


def cmd_report() -> None:
    gold, index = load_inputs()
    bm25 = bm25_ranking()
    report: dict[str, dict] = {"encoders": {}, "rerankers": {}}
    for path in sorted(OUT.glob("enc-*.json")):
        data = json.loads(path.read_text())
        dense = data["dense"]
        hybrid = {cid: rrf(dense[cid], bm25[cid]) for cid in dense}
        entry = {"revision": data["revision"], "device": data["device"],
                 "encode_s": data["encode_s"], "load_s": data["load_s"]}
        for split in ("all", "dev", "holdout"):
            g = gold if split == "all" else [c for c in gold if c["split"] == split]
            entry[f"dense[{split}]"] = metrics(dense, g, index)
            entry[f"hybrid_rrf60[{split}]"] = metrics(hybrid, g, index)
        report["encoders"][data["model"]] = entry
    for path in sorted(OUT.glob("rr-*.json")):
        data = json.loads(path.read_text())
        entry = {"revision": data["revision"], "device": data["device"],
                 "score_s": data["score_s"], "pairs": data["pairs"]}
        for enc, ranked in data["ranked"].items():
            for split in ("all", "holdout"):
                g = gold if split == "all" else [c for c in gold if c["split"] == split]
                entry[f"{enc}[{split}]"] = metrics(ranked, g, index)
        report["rerankers"][data["model"]] = entry
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "rerank", "report"])
    ap.add_argument("--model")
    ap.add_argument("--encoders", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cmd == "encode":
        cmd_encode(args.model, args.device, args.batch)
    elif args.cmd == "rerank":
        cmd_rerank(args.model, [e for e in args.encoders.split(",") if e], args.device,
                   args.batch)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
