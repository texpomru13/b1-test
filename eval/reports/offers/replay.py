"""Recalculate the offers report from the archived inputs, without network services."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[2]


def verify(data: bytes, expected: str, label: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 differs for {label}")


def main() -> None:
    hashes = json.loads((REPORT / "sha256.json").read_text())
    for name, expected in hashes["files"].items():
        verify((REPORT / name).read_bytes(), expected, name)
    manifest = json.loads((REPORT / "manifest.json").read_text())
    members = {"corpus.json", "gold.jsonl"}
    members.update(f"index/{fid}.json" for fid in manifest["inputs"]["documents"])
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(REPORT / "inputs.tar.gz") as archive:
        if set(archive.getnames()) != members or len(archive.getmembers()) != len(members):
            raise ValueError("unexpected contents in the input archive")
        work = Path(tmp).resolve()
        for name in sorted(members):
            member = archive.getmember(name)
            if not member.isfile() or not (work / name).resolve().is_relative_to(work):
                raise ValueError(f"invalid archived path {name}")
            entry = archive.extractfile(member)
            if entry is None:
                raise ValueError(f"missing file {name}")
            data = entry.read()
            verify(data, hashes["archive_members"][name], name)
            path = work / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
        run = work / "runs" / "final"
        run.mkdir(parents=True)
        for name in ("manifest.json", "responses.jsonl", "claim-review.json"):
            shutil.copyfile(REPORT / name, run / name)
        env = {**os.environ, "B1_EVAL_PACK": "external-six", "B1_EVAL_WORK_DIR": str(work)}
        for args in (("qualification_data.py", "--check-index"),
                     ("metrics.py", "--run", "final"),
                     ("qualification_review.py", "score", "--run", "final")):
            subprocess.run([sys.executable, REPO / "eval" / args[0], *args[1:]], cwd=REPO,
                           env=env, check=True)
        for name in ("cases-all.jsonl", "metrics-all.json", "qualification.json"):
            verify((run / name).read_bytes(), hashes["files"][name], name)
        shutil.copyfile(REPORT / "crosscheck-claims.json", run / "claim-review.json")
        subprocess.run([sys.executable, REPO / "eval" / "qualification_review.py",
                        "score", "--run", "final"], cwd=REPO, env=env, check=True)
        verify((run / "qualification.json").read_bytes(), hashes["files"]["crosscheck.json"],
               "crosscheck.json")
    print("Совпали исходные данные, технические метрики и результаты обеих смысловых оценок.")


if __name__ == "__main__":
    main()
