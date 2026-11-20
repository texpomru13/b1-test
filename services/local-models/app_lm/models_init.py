"""Provision the pinned encoder and reranker snapshots in the shared model cache.

Use a complete local snapshot or seed copy when available. Online mode can
download missing weights. On success, write models.lock.json in B1_RUNTIME_DIR
(the model cache by default)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path(os.environ.get("HF_HOME", "/models-cache"))
SEED = Path("/seed-cache")
RUNTIME = Path(os.environ.get("B1_RUNTIME_DIR", "/models-cache"))
OFFLINE = os.environ.get("OFFLINE", "0") == "1"
MAX_DOWNLOAD_ATTEMPTS = 3


def _snapshot_dir(root: Path, model_id: str, revision: str) -> Path:
    return root / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision


def _complete(snapshot: Path) -> bool:
    required = ["config.json", "tokenizer.json"]
    shard_index = snapshot / "model.safetensors.index.json"
    if shard_index.is_file():
        try:
            shards = json.loads(shard_index.read_text())["weight_map"].values()
            if not shards:
                return False
            required.extend(set(shards))
        except (ValueError, KeyError, AttributeError):
            return False
    else:
        required.append("model.safetensors")
    # The Giga encoders include locally loaded custom code and pooling modules.
    if "models--ai-sage--" in str(snapshot):
        required.extend(["configuration_gigarembed.py", "modeling_gigarembed.py",
                         "modules.json", "1_Pooling/config.json"])
    return all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
               for name in required)


def _present(model_id: str, revision: str) -> bool:
    return _complete(_snapshot_dir(CACHE, model_id, revision))


def _seed_copy(model_id: str, revision: str) -> bool:
    source = _snapshot_dir(SEED, model_id, revision)
    if not _complete(source):
        return False
    target = _snapshot_dir(CACHE, model_id, revision)
    print(f"[models-init] seeding {model_id}@{revision[:8]} from the host cache",
          flush=True)
    # Preserve other revisions and copy only the pinned snapshot, following its
    # links into the seed's shared blob store. Resume an interrupted seed copy.
    def copy_file(src: str, dst: str) -> str:
        path = Path(dst)
        if path.is_file() and path.stat().st_size == Path(src).stat().st_size:
            return dst
        tmp = path.with_suffix(path.suffix + ".part")
        shutil.copy2(src, tmp)
        tmp.replace(path)
        return dst

    shutil.copytree(source, target, dirs_exist_ok=True, copy_function=copy_file)
    return _present(model_id, revision)


def _download(model_id: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            print(f"[models-init] downloading {model_id}@{revision[:8]} "
                  f"(attempt {attempt})", flush=True)
            snapshot_download(model_id, revision=revision)  # resumes partial files
            return
        except Exception as exc:
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise
            delay = 5 * attempt
            print(f"[models-init] download failed ({type(exc).__name__}); "
                  f"retrying in {delay}s", flush=True)
            time.sleep(delay)


def main() -> int:
    assets = []
    encoder = os.environ.get("B1_EMBEDDING_MODEL", "")
    encoder_rev = os.environ.get("B1_EMBEDDING_REVISION", "")
    reranker = os.environ.get("B1_RERANKER_MODEL", "")
    reranker_rev = os.environ.get("B1_RERANKER_REVISION", "")
    if encoder and encoder_rev:
        assets.append((encoder, encoder_rev))
    if reranker and reranker_rev:
        assets.append((reranker, reranker_rev))
    if not assets:
        print("[models-init] nothing configured to provision", flush=True)
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    missing: list[str] = []
    for model_id, revision in assets:
        if _present(model_id, revision):
            print(f"[models-init] {model_id}@{revision[:8]} already present",
                  flush=True)
            continue
        if _seed_copy(model_id, revision):
            continue
        if OFFLINE:
            missing.append(f"{model_id}@{revision}")
            continue
        _download(model_id, revision)
        if not _present(model_id, revision):
            missing.append(f"{model_id}@{revision} (download incomplete)")

    if missing:
        print("[models-init] OFFLINE/incomplete: missing assets (no network "
              "fetch attempted):", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
        return 4

    lock = {
        "models": [
            {"id": model_id, "revision": revision,
             "snapshot": str(_snapshot_dir(CACHE, model_id, revision))}
            for model_id, revision in assets
        ],
    }
    lock_path = RUNTIME / "models.lock.json"
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    tmp.replace(lock_path)  # atomic manifest write
    subprocess.run(["chmod", "-R", "a+rX", str(CACHE)], check=False)
    print(f"[models-init] manifest written: {lock_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
