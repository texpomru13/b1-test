"""Shared locations for the corpus and results of an evaluation."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("B1_EVAL_WORK_DIR", ROOT / "eval" / "work")).resolve()


def gold_path() -> Path:
    """Use coordinates rebuilt for this index, or the published baseline for archived runs."""
    rebuilt = WORK / "gold.jsonl"
    return rebuilt if rebuilt.exists() else ROOT / "eval" / "gold" / "gold.jsonl"
