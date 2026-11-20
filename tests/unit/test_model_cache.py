"""An interrupted snapshot cannot satisfy a later offline launch."""

import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "services/local-models/app_lm/models_init.py"
spec = importlib.util.spec_from_file_location("models_init", MODULE)
assert spec is not None and spec.loader is not None
models_init = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_init)


def test_all_weight_shards_must_exist(tmp_path: Path) -> None:
    for name in ("config.json", "tokenizer.json", "one.safetensors"):
        (tmp_path / name).write_text("asset")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}'
    )
    assert not models_init._complete(tmp_path)
    (tmp_path / "two.safetensors").write_text("asset")
    assert models_init._complete(tmp_path)
