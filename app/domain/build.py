"""Content identity of the code and locked dependencies actually present in an image."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


def source_digest(root: Path, package: str) -> str:
    paths = [root / "pyproject.toml", root / "uv.lock", *sorted((root / package).rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def api_digest() -> str:
    return source_digest(Path(__file__).resolve().parents[2], "app")


if __name__ == "__main__":
    import sys

    print(source_digest(Path.cwd(), sys.argv[1]))
