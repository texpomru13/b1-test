#!/usr/bin/env python3
"""Pure profile resolver (DEPLOYMENT_SPEC §4, stage 8.2). STDLIB ONLY.

stdin:  JSON {"facts": {...}, "overrides": {...}}
stdout: JSON {"ok": bool, "decisions": {...}, "reasons": [...], "errors": [...]}

Runs identically on the host (unit tests import `resolve`) and inside the pinned
python base image (launcher pipes facts through `docker run`). The GLOBAL
decision is made exactly once per launch; the launcher materialises it into
.runtime/resolved.env BEFORE any Compose interpolation.

Rules implemented (plan §3.2/§4.2-4.3, ADR 002/005):
- Priority CLI override > saved/explicit env > existing index fingerprint > auto.
- CPU alone NEVER downgrades quality; only a verified memory deficit does, and
  the result carries quality_degraded=true + a human reason.
- Mac auto order: gated DMR (gate NOT passed yet -> skipped with a reason) ->
  host Ollama (verified reachable) -> container CPU.
- NVIDIA only after a positive container smoke fact; forced nvidia without the
  fact is an ERROR, never a silent substitution.
- --strict-profile / explicit models: impossible fit -> error, no substitution.
- The existing index fingerprint constrains the encoder: a conflicting explicit
  encoder choice is an error pointing to a controlled reindex.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from typing import Any

# Memory models (GiB, fp32/Q4 artifacts + runtime headroom; starting estimates
# from measured stage 0-6 values: Giga3B 14.4 RSS, 480M 3.5, reranker8B ~32 fp32
# est / measured container 33.0 for both resident, 27B Q4 ~17 artifact + KV).
ENCODERS = {
    "ai-sage/Giga-Embeddings-instruct": {
        "revision": "2cf0fdc97194aaedf10ac0e6bf798834acd31042",
        "dim": 2048, "rss_gib": 14.5, "quality_rank": 0,
    },
    "ai-sage/Giga-Embeddings-instruct-480M-0826": {
        "revision": "1763d603adac8057bd6482a708001b0ed2e0a903",
        "dim": 1024, "rss_gib": 3.6, "quality_rank": 1,
    },
}
RERANKERS = {
    "Qwen/Qwen3-Reranker-8B": {
        "revision": "77d193c791ed757ca307ee72715aa132723da912",
        "adapter": "qwen_yes_no", "rss_gib": 19.0, "quality_rank": 0,
    },
    "BAAI/bge-reranker-v2-m3": {
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "adapter": "cross_encoder", "rss_gib": 2.2, "quality_rank": 1,
    },
}
LLMS = {
    "qwen3.6:27b-q4_K_M": {"rss_gib": 20.0, "quality_rank": 0, "dmr_id": "ai/qwen3.6:27B"},
    # Explicit GGUF quantization tags: on Apple Silicon Ollama >= 0.40 resolves the
    # bare "qwen3.5:9b" tag to an MLX (nvfp4) artifact whose runner rejects
    # structured output ("structured output is unavailable", stage-9 finding), and
    # the bare tag is a different artifact per platform. Q4_K_M runs on llama.cpp
    # everywhere and supports the JSON-schema `format` the QA graph requires.
    # thinking=off: measured on the stage-9 DEV split (eval/reports): with thinking
    # the 9B model spends num_predict on reasoning and returns empty JSON (5/24 502s,
    # p50 145 s) vs 0 errors and p50 19 s without it. LLM_THINKING overrides.
    "qwen3.5:9b-q4_K_M": {"rss_gib": 7.5, "quality_rank": 1, "dmr_id": None,
                          "thinking": "off"},
    "qwen3.5:4b-q4_K_M": {"rss_gib": 3.5, "quality_rank": 2, "dmr_id": None,
                          "thinking": "off"},
}
INFRA_RESERVE_GIB = 2.5   # redis + qdrant + api/worker/dispatcher
SAFETY_RATIO = 0.8        # plan: keep a 20-30% reserve on top of the reserve

PROFILES = {
    "quality": {"encoder": "ai-sage/Giga-Embeddings-instruct",
                "reranker": "Qwen/Qwen3-Reranker-8B", "llm": "qwen3.6:27b-q4_K_M"},
    "quality-compact": {"encoder": "ai-sage/Giga-Embeddings-instruct",
                        "reranker": "BAAI/bge-reranker-v2-m3",
                        "llm": "qwen3.6:27b-q4_K_M"},
    "balanced": {"encoder": "ai-sage/Giga-Embeddings-instruct-480M-0826",
                 "reranker": "BAAI/bge-reranker-v2-m3", "llm": "qwen3.5:9b-q4_K_M"},
    "low-memory": {"encoder": "ai-sage/Giga-Embeddings-instruct-480M-0826",
                   "reranker": "BAAI/bge-reranker-v2-m3", "llm": "qwen3.5:4b-q4_K_M"},
}
PROFILE_ORDER = ["quality", "quality-compact", "balanced", "low-memory"]


def _is_split_pools(facts: dict, llm_runtime: str) -> bool:
    """Mac: the Docker VM is pre-carved from host RAM, and a host-side LLM runtime
    (host Ollama / DMR) uses the remaining host memory -> two separate pools.
    Native Linux (or an LLM container next to retrieval): ONE shared pool —
    unified/host memory is never summed with the VM (plan §4.3)."""
    return facts.get("os") == "darwin" and llm_runtime in ("host-ollama", "dmr")


def _retrieval_pool_gib(facts: dict) -> float:
    return (facts.get("docker_vm_mem_gib", 0.0) - INFRA_RESERVE_GIB) * SAFETY_RATIO


def _host_pool_gib(facts: dict) -> float:
    return facts.get("host_free_ram_gib", 0.0) * SAFETY_RATIO


def _pick_llm_runtime(facts: dict, overrides: dict, reasons: list[str],
                      errors: list[str]) -> str | None:
    provider = overrides.get("provider") or "auto"
    saved = overrides.get("saved_runtime") or ""
    device = overrides.get("device") or "auto"
    if provider not in ("auto", "ollama", "dmr", "openai", "anthropic"):
        errors.append(f"unknown LLM provider '{provider}'")
        return None
    if device not in ("auto", "cpu", "metal", "nvidia"):
        errors.append(f"unknown device '{device}'")
        return None
    if overrides.get("offline"):
        use_host = provider == "ollama" or saved == "host-ollama" or device == "metal"
        if (use_host and provider in ("auto", "ollama") and device in ("auto", "metal")
                and facts.get("os") == "darwin" and facts.get("host_ollama_reachable")):
            return "host-ollama"
        if provider not in ("auto", "ollama") or device == "metal":
            errors.append("offline mode requires the containerized Ollama runtime")
            return None
        return "container-ollama-gpu" if device == "nvidia" else "container-ollama"
    if device != "auto":
        if provider in ("openai", "anthropic", "dmr"):
            errors.append(f"--device {device} requires a local Ollama provider")
            return None
        if device == "cpu":
            return "container-ollama"
        if device == "nvidia":
            return "container-ollama-gpu"
        if (facts.get("os") == "darwin" and facts.get("arch") == "arm64"
                and facts.get("host_ollama_reachable")):
            return "host-ollama"
        errors.append("--device metal requires reachable host Ollama on Apple Silicon")
        return None
    if provider == "auto" and saved:
        # A saved valid profile NEVER migrates silently when a new runtime appears
        # (DEPLOYMENT_SPEC §3); an explicit --provider is the only way to switch.
        still_valid = (
            (saved == "host-ollama" and facts.get("host_ollama_reachable"))
            or (saved == "dmr" and facts.get("dmr_running"))
            or saved in ("container-ollama",)
            or (saved == "container-ollama-gpu"
                and facts.get("nvidia_container_smoke_ok"))
        )
        if still_valid:
            reasons.append(f"saved profile kept: runtime={saved} (no silent "
                           "migration; use --provider to switch)")
            return saved
        reasons.append(f"saved runtime '{saved}' is no longer available -> "
                       "re-resolving automatically")
    if provider == "openai" or provider == "anthropic":
        return provider
    if provider == "dmr":
        if not facts.get("dmr_running"):
            errors.append("provider=dmr requested but Docker Model Runner is not "
                          "running; enable it in Docker Desktop")
            return None
        reasons.append("provider=dmr chosen explicitly (DMR gate: mechanics passed; "
                       "auto-preference stays blocked per ADR 005)")
        return "dmr"
    if provider == "ollama":
        if facts.get("os") == "darwin" and facts.get("host_ollama_reachable"):
            reasons.append("provider=ollama: verified host Ollama endpoint")
            return "host-ollama"
        if facts.get("nvidia_container_smoke_ok"):
            return "container-ollama-gpu"
        reasons.append("provider=ollama: using the container CPU runtime")
        return "container-ollama"
    # auto
    if facts.get("os") == "darwin":
        if facts.get("dmr_running") and facts.get("dmr_gate_passed"):
            reasons.append("mac auto: DMR available and gate passed")
            return "dmr"
        if facts.get("dmr_running"):
            reasons.append("mac auto: DMR present but its compatibility gate is not "
                           "fully passed (ADR 005) -> not auto-preferred")
        if facts.get("host_ollama_reachable"):
            reasons.append("mac auto: verified host Ollama (Metal)")
            return "host-ollama"
        reasons.append("mac auto: no accelerated runtime verified -> container CPU")
        return "container-ollama"
    if facts.get("nvidia_container_smoke_ok"):
        reasons.append("linux auto: NVIDIA container smoke passed")
        return "container-ollama-gpu"
    if facts.get("nvidia_smi_present") and not facts.get("nvidia_container_smoke_ok"):
        reasons.append("nvidia-smi present but container smoke NOT passed -> CPU "
                       "(no silent GPU enablement)")
    return "container-ollama"


def resolve(facts: dict, overrides: dict) -> dict[str, Any]:
    reasons: list[str] = []
    errors: list[str] = []
    strict = bool(overrides.get("strict_profile"))

    if overrides.get("device") == "nvidia" and not facts.get("nvidia_container_smoke_ok"):
        errors.append("device=nvidia forced, but the container GPU smoke has not "
                      "passed on this host — refusing (no hidden CPU substitution); "
                      "run with --device auto for an explained CPU fallback")
        return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}

    llm_runtime = _pick_llm_runtime(facts, overrides, reasons, errors)
    if errors:
        return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}
    cloud = llm_runtime in ("openai", "anthropic")


    requested_profile = overrides.get("quality") or "balanced"
    if cloud and not overrides.get("model"):
        return {"ok": False, "decisions": {}, "reasons": reasons,
                "errors": ["a cloud provider requires an explicit --model or LLM_MODEL"]}
    if requested_profile not in PROFILES:
        return {"ok": False, "decisions": {}, "reasons": reasons,
                "errors": [f"unknown quality profile '{requested_profile}'"]}
    start_index = PROFILE_ORDER.index(requested_profile)

    explicit_encoder = overrides.get("embedding_model")
    explicit_reranker = overrides.get("reranker_model")
    # In cloud mode --model names the CLOUD model and never constrains the local
    # retrieval profile selection.
    explicit_llm = None if cloud else overrides.get("model")

    existing_fp_encoder = facts.get("existing_index_encoder")  # from index fingerprint
    if existing_fp_encoder and explicit_encoder and explicit_encoder != existing_fp_encoder:
        errors.append(
            f"existing index was built with encoder '{existing_fp_encoder}'; "
            f"explicit '{explicit_encoder}' conflicts — run a controlled reindex "
            "explicitly instead of changing the embedding space in place")
        return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}

    chosen = None
    degraded_reason = None
    for profile_name in PROFILE_ORDER[start_index:]:
        profile = dict(PROFILES[profile_name])
        if existing_fp_encoder:
            profile["encoder"] = existing_fp_encoder
        if explicit_encoder:
            profile["encoder"] = explicit_encoder
        if explicit_reranker:
            profile["reranker"] = explicit_reranker
        if explicit_llm:
            profile["llm"] = explicit_llm
        if profile["encoder"] not in ENCODERS or profile["reranker"] not in RERANKERS:
            errors.append(f"unknown model in profile: {profile}")
            return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}
        enc_gib = ENCODERS[profile["encoder"]]["rss_gib"]
        rr_gib = RERANKERS[profile["reranker"]]["rss_gib"]
        llm_model = profile["llm"]
        llm_gib = 0.0
        if not cloud:
            if llm_model in LLMS:
                llm_gib = LLMS[llm_model]["rss_gib"]
            elif explicit_llm:
                reasons.append(f"explicit LLM '{llm_model}': no memory model known; "
                               "fit must be confirmed at warmup")
            else:
                errors.append(f"unknown LLM '{llm_model}'")
                return {"ok": False, "decisions": {}, "reasons": reasons,
                        "errors": errors}

        # local-models keeps both weight sets resident for its entire lifetime.
        retrieval_need = enc_gib + rr_gib
        if _is_split_pools(facts, llm_runtime):
            retrieval_pool = _retrieval_pool_gib(facts)
            llm_pool = _host_pool_gib(facts)
            fits = retrieval_need <= retrieval_pool and llm_gib <= llm_pool
            need_desc = (f"retrieval {retrieval_need:.1f} GiB vs VM pool "
                         f"{max(retrieval_pool, 0):.1f} / LLM {llm_gib:.1f} vs "
                         f"host pool {max(llm_pool, 0):.1f} GiB")
        else:
            base_pool = (facts.get("docker_vm_mem_gib", 0.0) if facts.get("os") == "darwin"
                         else facts.get("host_free_ram_gib", 0.0))
            pool = (base_pool - INFRA_RESERVE_GIB) * SAFETY_RATIO
            need = llm_gib + retrieval_need
            fits = need <= pool
            need_desc = f"resident {need:.1f} GiB vs shared pool {max(pool, 0):.1f} GiB"

        if fits:
            chosen = (profile_name, profile)
            break
        if strict:
            errors.append(f"profile '{profile_name}' does not fit ({need_desc}); "
                          "substitution is forbidden by --strict-profile")
            return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}
        degraded_reason = f"verified memory deficit: profile '{profile_name}' needs {need_desc}"
        reasons.append(degraded_reason + " -> trying the next profile")

    if chosen is None:
        errors.append("no profile fits the verified memory budget; raise the Docker "
                      "Desktop memory limit or free host RAM")
        return {"ok": False, "decisions": {}, "reasons": reasons, "errors": errors}

    profile_name, profile = chosen
    model = overrides.get("model") if cloud else profile["llm"]
    if llm_runtime == "dmr":
        # Prefer a user-supplied DMR ID, or translate the Ollama preset when known.
        model = LLMS.get(model, {}).get("dmr_id") or overrides.get("model")
        if not model:
            return {"ok": False, "decisions": {}, "reasons": reasons,
                    "errors": ["specify an installed DMR --model for this profile"]}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@+-]*", model):
        return {"ok": False, "decisions": {}, "reasons": reasons,
                "errors": ["invalid LLM model identifier"]}
    degraded = profile_name != requested_profile
    encoder = ENCODERS[profile["encoder"]]
    reranker = RERANKERS[profile["reranker"]]
    decisions = {
        "profile": profile_name,
        "preset_status": "candidate",
        "llm_runtime": llm_runtime,
        "llm_provider": ("ollama" if llm_runtime in ("host-ollama", "container-ollama",
                                                     "container-ollama-gpu")
                         else llm_runtime),
        "llm_model": model,
        "dmr_model": model if llm_runtime == "dmr" else None,
        # model-level default reasoning mode for Ollama (env LLM_THINKING wins)
        "llm_thinking": ("auto" if cloud
                         else (LLMS.get(profile["llm"], {}) or {}).get("thinking", "auto")),
        "host_ollama_port": facts.get("host_ollama_port", 11434),
        "ollama_base_url": (f"http://host.docker.internal:{facts.get('host_ollama_port', 11434)}"
                            if llm_runtime == "host-ollama"
                            else "http://ollama:11434"),
        "embedding_model": profile["encoder"],
        "embedding_revision": encoder["revision"],
        "embedding_dim": encoder["dim"],
        "reranker_model": profile["reranker"],
        "reranker_revision": reranker["revision"],
        "reranker_adapter": reranker["adapter"],
        "model_residency": "resident-both",
        "quality_degraded": degraded,
        "quality_degraded_reason": degraded_reason if degraded else None,
        "use_ollama_container": llm_runtime in ("container-ollama",
                                                "container-ollama-gpu"),
        "compose_overrides": (
            ["compose.mac.yaml"] if llm_runtime == "host-ollama"
            else ["compose.dmr.yaml"] if llm_runtime == "dmr"
            else ["compose.gpu.yaml"] if llm_runtime == "container-ollama-gpu"
            else []
        ),
        "device": overrides.get("device") or "auto",
        "accelerator_verified": llm_runtime in ("host-ollama", "dmr",
                                                "container-ollama-gpu"),
        # §8.6: slow-CPU profile gets the long (but finite) deadlines
        "llm_read_timeout_s": (
            300 if llm_runtime in ("host-ollama", "dmr", "container-ollama-gpu",
                                   "openai", "anthropic") else 1200
        ),
        "qa_deadline_s": (
            900 if llm_runtime in ("host-ollama", "dmr", "container-ollama-gpu")
            else 420 if llm_runtime in ("openai", "anthropic") else 3600
        ),
    }
    if cloud:
        decisions["use_ollama_container"] = False
        decisions["compose_overrides"] = ["compose.cloud.yaml"]
        reasons.append("cloud profile: NO local LLM is started or downloaded; "
                       "retrieval and reranker stay local")
    reasons.append(
        f"resolved profile={profile_name} runtime={llm_runtime} "
        f"encoder={profile['encoder']} reranker={profile['reranker']} "
        "residency=resident-both")
    return {"ok": True, "decisions": decisions, "reasons": reasons, "errors": []}


def shell_environment(result: dict, args: argparse.Namespace) -> str:
    d = result["decisions"]
    encoder = d["embedding_model"].replace("/", "--")
    revision = d["embedding_revision"]
    variables = {
        "B1_SAVED_PROVIDER": args.provider, "B1_SAVED_RUNTIME": d["llm_runtime"],
        "B1_SAVED_MODEL": args.model, "B1_SAVED_QUALITY": args.quality,
        "B1_SAVED_RERANKER": args.reranker_model, "B1_SAVED_DEVICE": args.device,
        "B1_API_PORT": args.port, "B1_HOST_OLLAMA_PORT": d["host_ollama_port"],
        "B1_HF_SEED_DIR": args.seed_dir, "B1_OFFLINE": int(args.offline),
        "B1_QUALITY_DEGRADED": int(d["quality_degraded"]),
        "B1_QUALITY_DEGRADED_REASON": d["quality_degraded_reason"] or "",
        "B1_TOKENIZER_PATH": (f"/models-cache/hub/models--{encoder}/snapshots/{revision}/"
                              "tokenizer.json"),
        "B1_LLM_THINKING": d["llm_thinking"],
    }
    for key in ("llm_provider", "llm_model", "ollama_base_url", "embedding_model",
                "embedding_revision", "reranker_model", "reranker_revision",
                "reranker_adapter", "llm_read_timeout_s", "qa_deadline_s"):
        variables[f"B1_{key.upper()}"] = d[key]
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in variables.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("result", "doctor", "env", "summary"),
                        default="result")
    for option in ("provider", "model", "device", "quality", "embedding-model", "reranker-model",
                   "saved-runtime", "saved-encoder", "seed-dir", "port"):
        parser.add_argument("--" + option, default="")
    parser.add_argument("--strict-profile", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    data = json.load(sys.stdin)
    if args.format in ("env", "summary"):
        if args.format == "env":
            print(shell_environment(data, args))
        else:
            for reason in data["reasons"] + data["errors"]:
                print("    -", reason)
        return 0
    # Keep the standalone JSON interface used by earlier callers.
    facts = data.get("facts", data)
    if args.saved_encoder:
        facts["existing_index_encoder"] = args.saved_encoder
    result = resolve(facts, data.get("overrides", vars(args)))
    output = {"facts": facts, "resolution": result} if args.format == "doctor" else result
    print(json.dumps(output, ensure_ascii=False, indent=1))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
