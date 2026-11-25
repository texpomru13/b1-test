"""Stage-8.4 unit matrix for the pure resolver (DEPLOYMENT_SPEC §8 item 3).

These are RESOLVER tests with mocked facts — per the plan they never count as
real hardware smokes (those are recorded separately with not_run marks).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from resolver import resolve


def mac_facts(host_free: float, vm: float, **extra: Any) -> dict:
    return {"os": "darwin", "arch": "arm64", "host_free_ram_gib": host_free,
            "docker_vm_mem_gib": vm, "host_ollama_reachable": True,
            "dmr_running": True, "dmr_gate_passed": False, **extra}


def linux_cpu_facts(ram: float, **extra: Any) -> dict:
    return {"os": "linux", "arch": "x86_64", "host_free_ram_gib": ram,
            "docker_vm_mem_gib": ram,  # native linux: no VM split
            "host_ollama_reachable": False, "dmr_running": False, **extra}


# --- CPU rows: 12/16/32/64 GB ---


def test_cpu_large_pool_keeps_explicit_27b_quality() -> None:
    result = resolve(linux_cpu_facts(80.0), {"quality": "quality"})
    assert result["ok"], result
    d = result["decisions"]
    assert d["profile"] == "quality"
    assert d["llm_model"] == "qwen3.6:27b-q4_K_M"  # CPU never downgrades by itself
    assert d["quality_degraded"] is False
    assert d["use_ollama_container"] is True
    assert d["accelerator_verified"] is False  # honest: no accelerator on CPU


def test_cpu_32gb_degrades_with_visible_reason() -> None:
    result = resolve(linux_cpu_facts(32.0), {"quality": "quality"})
    assert result["ok"]
    d = result["decisions"]
    assert d["profile"] != "quality"
    assert d["quality_degraded"] is True
    assert "memory deficit" in (d["quality_degraded_reason"] or "")


def test_cpu_16gb_and_12gb_pick_smaller_presets() -> None:
    d16 = resolve(linux_cpu_facts(16.0), {})["decisions"]
    assert d16["quality_degraded"] is True
    assert d16["embedding_model"].endswith("480M-0826")
    d12 = resolve(linux_cpu_facts(12.0), {})["decisions"]
    assert not d12  # even the resident 4B stack exceeds the measured pool


def test_cpu_8gb_refuses_instead_of_pretending() -> None:
    result = resolve(linux_cpu_facts(8.0), {})
    assert result["ok"] is False
    assert any("no profile fits" in e for e in result["errors"])


# --- Mac rows: 36/48/64 GB host with a constrained Docker VM ---


def test_mac_64_host_prefers_host_ollama_dmr_gate_blocked() -> None:
    result = resolve(mac_facts(host_free=64.0, vm=48.0), {})
    assert result["ok"]
    d = result["decisions"]
    assert d["llm_runtime"] == "host-ollama"  # DMR present but gate not passed
    assert d["ollama_base_url"] == "http://host.docker.internal:11434"
    assert d["compose_overrides"] == ["compose.mac.yaml"]
    assert any("gate is not fully passed" in r for r in result["reasons"])
    assert d["accelerator_verified"] is True


def test_mac_small_vm_requires_a_smaller_resident_profile() -> None:
    # VM fits each weight alone (14.5 / 19.0) but not both (33.5): 30*0.75-3=19.5
    result = resolve(mac_facts(host_free=48.0, vm=30.0), {"quality": "quality"})
    assert result["ok"]
    d = result["decisions"]
    assert d["profile"] == "quality-compact"
    assert d["model_residency"] == "resident-both"


def test_mac_36_vm_too_small_degrades() -> None:
    result = resolve(mac_facts(host_free=36.0, vm=12.0), {"quality": "quality"})
    assert result["ok"]
    assert result["decisions"]["quality_degraded"] is True


def test_mac_dmr_gate_passed_auto_prefers_dmr() -> None:
    result = resolve(mac_facts(64.0, 48.0, dmr_gate_passed=True), {"quality": "quality"})
    d = result["decisions"]
    assert d["llm_runtime"] == "dmr"
    assert d["dmr_model"] == "ai/qwen3.6:27B"
    assert d["compose_overrides"] == ["compose.dmr.yaml"]


# --- GPU rows ---


def test_nvidia_smoke_ok_enables_gpu_profile() -> None:
    result = resolve(linux_cpu_facts(64.0, nvidia_smi_present=True,
                                     nvidia_container_smoke_ok=True), {})
    d = result["decisions"]
    assert d["llm_runtime"] == "container-ollama-gpu"
    assert d["compose_overrides"] == ["compose.gpu.yaml"]


def test_nvidia_present_but_broken_falls_back_with_reason() -> None:
    result = resolve(linux_cpu_facts(64.0, nvidia_smi_present=True,
                                     nvidia_container_smoke_ok=False), {})
    assert result["decisions"]["llm_runtime"] == "container-ollama"
    assert any("smoke NOT passed" in r for r in result["reasons"])


def test_forced_nvidia_without_smoke_is_an_error() -> None:
    result = resolve(linux_cpu_facts(64.0, nvidia_smi_present=True),
                     {"device": "nvidia"})
    assert result["ok"] is False
    assert any("refusing" in e for e in result["errors"])


# --- overrides / strict / fingerprint ---


def test_strict_profile_errors_instead_of_substituting() -> None:
    result = resolve(linux_cpu_facts(16.0), {"strict_profile": True})
    assert result["ok"] is False
    assert any("substitution is forbidden" in e for e in result["errors"])


def test_explicit_model_never_silently_replaced() -> None:
    result = resolve(linux_cpu_facts(12.0),
                     {"embedding_model": "ai-sage/Giga-Embeddings-instruct"})
    assert result["ok"] is False  # 3B cannot fit, and substitution is forbidden


def test_existing_fingerprint_constrains_encoder() -> None:
    facts = linux_cpu_facts(
        64.0, existing_index_encoder="ai-sage/Giga-Embeddings-instruct")
    conflict = resolve(facts, {
        "embedding_model": "ai-sage/Giga-Embeddings-instruct-480M-0826"})
    assert conflict["ok"] is False
    assert any("controlled reindex" in e for e in conflict["errors"])
    keep = resolve(facts, {})
    assert keep["decisions"]["embedding_model"] == "ai-sage/Giga-Embeddings-instruct"


def test_explicit_dmr_requires_running_runtime() -> None:
    result = resolve(mac_facts(64.0, 48.0, dmr_running=False), {"provider": "dmr"})
    assert result["ok"] is False


def test_cloud_profile_starts_no_local_llm() -> None:
    result = resolve(linux_cpu_facts(16.0), {"provider": "openai",
                                             "model": "gpt-6.1-sol"})
    assert result["ok"]
    d = result["decisions"]
    assert d["llm_runtime"] == "openai"
    assert d["use_ollama_container"] is False
    assert d["compose_overrides"] == ["compose.cloud.yaml"]
    # retrieval stays LOCAL even in cloud mode (16GB -> compact encoder fits)
    assert d["embedding_model"] in (
        "ai-sage/Giga-Embeddings-instruct", "ai-sage/Giga-Embeddings-instruct-480M-0826")


def test_saved_profile_never_migrates_silently() -> None:
    # container profile was saved; host Ollama appeared later -> STAYS container
    facts = mac_facts(host_free=64.0, vm=48.0)  # host ollama reachable now
    kept = resolve(facts, {"saved_runtime": "container-ollama"})
    assert kept["decisions"]["llm_runtime"] == "container-ollama"
    assert any("no silent migration" in r for r in kept["reasons"])
    # explicit --provider switches
    switched = resolve(facts, {"provider": "ollama",
                               "saved_runtime": "container-ollama"})
    assert switched["decisions"]["llm_runtime"] == "host-ollama"
    # saved runtime that disappeared re-resolves with a reason
    gone = resolve(mac_facts(64.0, 48.0, host_ollama_reachable=False),
                   {"saved_runtime": "host-ollama"})
    assert gone["decisions"]["llm_runtime"] == "container-ollama"
    assert any("no longer available" in r for r in gone["reasons"])


def test_mac_container_llm_uses_vm_pool_not_host_free() -> None:
    # host RAM is busy (models elsewhere), but the 63GiB VM holds everything
    facts = mac_facts(host_free=30.0, vm=62.9, host_ollama_reachable=False)
    result = resolve(facts, {"quality": "quality", "saved_runtime": "container-ollama"})
    assert result["ok"], result
    d = result["decisions"]
    assert d["llm_runtime"] == "container-ollama"
    assert d["profile"] == "quality-compact"  # both weights must actually fit


def test_llm_thinking_default_follows_model() -> None:
    quality = resolve(linux_cpu_facts(80.0), {"quality": "quality"})["decisions"]
    assert quality["llm_thinking"] == "auto"  # no stage-9 evidence for the 27B model
    for model in ("qwen3.5:9b-q4_K_M", "qwen3.5:4b-q4_K_M"):
        d = resolve(mac_facts(64.0, 63.0), {"model": model})["decisions"]
        assert d["llm_model"] == model
        assert d["llm_thinking"] == "off"

def test_default_is_measured_balanced_profile() -> None:
    assert resolve(linux_cpu_facts(64.0), {})["decisions"]["profile"] == "balanced"


def test_explicit_cpu_and_offline_select_container_on_mac() -> None:
    for override in ({"device": "cpu"}, {"offline": True}):
        decisions = resolve(mac_facts(64, 48), override)["decisions"]
        assert decisions["llm_runtime"] == "container-ollama"


def test_invalid_runtime_or_device_is_rejected() -> None:
    for override in ({"provider": "typo"}, {"device": "typo"}, {"device": "metal"}):
        assert not resolve(linux_cpu_facts(64), override)["ok"]


def test_dmr_uses_actual_installed_model_identifier() -> None:
    result = resolve(mac_facts(64, 48), {"provider": "dmr", "model": "ai/test:9b"})
    assert result["decisions"]["llm_model"] == "ai/test:9b"


def test_cloud_requires_explicit_model() -> None:
    assert not resolve(linux_cpu_facts(64), {"provider": "openai"})["ok"]


def test_explicit_host_ollama_can_run_offline() -> None:
    result = resolve(mac_facts(64, 48), {"provider": "ollama", "offline": True})
    assert result["decisions"]["llm_runtime"] == "host-ollama"


def test_openai_full_is_explicit_and_does_not_reuse_a_local_index() -> None:
    options = {"provider": "openai-full"}
    result = resolve(linux_cpu_facts(6), options)
    assert result["ok"]
    assert result["decisions"]["llm_model"] == "gpt-6.2-sol"
    assert result["decisions"]["embedding_model"] == "text-embedding-3-large"
    assert not resolve(linux_cpu_facts(6), {**options, "offline": True})["ok"]
    local_index = linux_cpu_facts(16, existing_index_encoder="ai-sage/Giga-Embeddings-instruct")
    assert not resolve(local_index, options)["ok"]
