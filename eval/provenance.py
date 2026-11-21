"""Bind measurements to the live API, the published index and, for Compose, its images."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import httpx
from paths import ROOT, WORK, gold_path

sys.path.insert(0, str(ROOT))
from app.domain.build import source_digest

# Read effective settings inside the API process's container. Export only known
# non-secret QA settings. URLs may contain credentials and never enter a report.
MODEL_CONFIG = """
import json
from app.config.settings import get_settings
import httpx
settings = get_settings()
keys = '''llm_provider llm_model llm_temperature llm_thinking local_context_tokens
llm_read_timeout_s qa_deadline_s llm_max_output_tokens llm_reasoning_budget_tokens
embedding_model embedding_instruction_version chunk_tokens chunk_overlap_tokens
dense_top_k sparse_top_k fusion_method rerank_input_k context_top_k context_budget_tokens
bm25_language bm25_avg_len max_repair_attempts max_llm_calls_per_question llm_max_attempts
max_concurrent_qa qdrant_collection'''.split()
data = {"settings": settings.model_dump(mode="json", include=set(keys))}
health = httpx.get(settings.local_models_url.rstrip("/") + "/v1/health", timeout=10)
health.raise_for_status()
models = health.json()
assert models["status"] == "ok", "model service is not ready"
data["local_models"] = {k: models[k] for k in (
    "loaded_model", "revision", "reranker_model", "reranker_revision", "reranker_adapter", "device"
)}
if settings.llm_provider in ("auto", "ollama"):
    tags = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=10)
    tags.raise_for_status()
    name = settings.llm_model
    data["llm_digest"] = next(m["digest"] for m in tags.json()["models"]
                              if m["name"] in (name, name + ":latest"))
print(json.dumps(data))
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(args, input=stdin, cwd=ROOT, text=True, check=True,
                          capture_output=True, timeout=60).stdout.strip()


def docker_id(project: str, service: str) -> str:
    result = command("docker", "ps", "--no-trunc", "-q",
                     "--filter", f"label=com.docker.compose.project={project}",
                     "--filter", f"label=com.docker.compose.service={service}").splitlines()
    if len(result) != 1:
        raise ValueError(f"expected one running {service} container in project {project}")
    return result[0]


def compose_runtime(project: str, instance: str) -> dict:
    images = {}
    source_file = (ROOT / "app/domain/build.py").read_text()
    for name in ("api", "worker", "dispatcher", "local-models", "redis", "qdrant"):
        cid = docker_id(project, name)
        image = command("docker", "inspect", "-f", "{{.Image}}", cid)
        images[name] = {"image_id": image}
        if name == "api" and command("docker", "exec", cid, "hostname") != instance:
            raise ValueError("--api reached a different container than --project")
        if name in ("redis", "qdrant"):
            continue
        is_model_service = name == "local-models"
        package = "app_lm" if is_model_service else "app"
        root = ROOT / "services/local-models" if is_model_service else ROOT
        actual = command("docker", "exec", "-i", cid, "python", "-", package, stdin=source_file)
        if actual != source_digest(root, package):
            raise ValueError(f"stale {name} build; rebuild the project before measuring")
        images[name]["source_sha256"] = actual
    api = docker_id(project, "api")
    runtime = json.loads(command("docker", "exec", "-i", api, "python", "-", stdin=MODEL_CONFIG))
    model_container = docker_id(project, "local-models")
    for key in ("LM_BATCH_SIZE", "LM_RERANKER_BATCH"):
        runtime["local_models"][key] = command("docker", "exec", model_container, "printenv", key)
    return {"project": project, "images": images, **runtime}


def live_inputs(client: httpx.Client, project: str | None, doc_ids: dict[str, str]) -> dict:
    build = client.get("/health/build")
    build.raise_for_status()
    identity = build.json()
    code = source_digest(ROOT, "app")
    if identity["source_sha256"] != code:
        raise ValueError("the responding API and the local checkout have different source hashes")
    inputs = {}
    corpus = json.loads((WORK / "corpus.json").read_text())["docs"]
    for fid, doc_id in doc_ids.items():
        index = json.loads((WORK / "index" / f"{fid}.json").read_text())
        detail = client.get(f"/api/v1/documents/{doc_id}")
        detail.raise_for_status()
        doc = detail.json()
        digest = sha256(ROOT / corpus[fid]["file"])
        if (not doc["ready"] or digest != doc.get("source_sha256")
                or not doc.get("index_fingerprint") or not doc.get("accepted_run_id")
                or any(doc[k] != index[k] for k in ("accepted_run_id", "index_fingerprint"))):
            raise ValueError(f"document or index for {fid} differs from the live service")
        inputs[fid] = {"doc_id": doc_id, "source_sha256": digest,
                       "index_fingerprint": doc["index_fingerprint"],
                       "accepted_run_id": doc["accepted_run_id"],
                       "export_sha256": sha256(WORK / "index" / f"{fid}.json")}
    return {
        "api_source_sha256": code, "documents": inputs, "gold_sha256": sha256(gold_path()),
        "runtime": compose_runtime(project, identity["instance"]) if project else {
            "instance": identity["instance"], "image_inspection": "not requested"},
    }


def environment() -> dict:
    hardware = {"platform": platform.platform(), "arch": platform.machine(),
                "cpu": platform.processor()}
    if sys.platform == "darwin":
        hardware.update(cpu=command("sysctl", "-n", "machdep.cpu.brand_string"),
                        memory_bytes=int(command("sysctl", "-n", "hw.memsize")))
    elif sys.platform.startswith("linux"):
        memory = Path("/proc/meminfo").read_text().splitlines()[0].split()[1]
        hardware["memory_bytes"] = int(memory) * 1024
    return {"git_commit": command("git", "rev-parse", "HEAD"),
            "git_dirty": bool(command("git", "status", "--porcelain")), "hardware": hardware}


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
