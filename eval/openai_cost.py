"""Calculate Standard API charges from OpenAI's reported per-call usage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Per million tokens: ordinary input, cached input, output. Cache writes on
# GPT-6 use 1.25x for 30 minutes and 2.5x for 12 hours.
STANDARD = {
    "gpt-6.2-sol": (2.0, 0.10, 10.0),
    "gpt-6.1-terra": (1.0, 0.05, 5.0),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "text-embedding-3-large": (0.13, 0.0, 0.0),
}


def cost(row: dict) -> float:
    name = row["model"]
    model = next((m for m in STANDARD if name == m or name.startswith(m + "-")), None)
    if model is None or row["service_tier"] not in (None, "default", "standard"):
        raise ValueError(f"no Standard rate for {name} / {row['service_tier']}")
    rate_in, rate_cached, rate_out = STANDARD[model]
    if model.startswith("gpt-6.") and row["input_tokens"] > 272_000:
        rate_in *= 2
        rate_cached *= 2
        rate_out *= 1.5
    details = row["input_details"]
    cached = details.get("cached_tokens", 0)
    written = details.get("cache_write_tokens", 0)
    long_write = (details.get("cache_write_tokens_details") or {}).get("ttl_12h_tokens", 0)
    uncached = row["input_tokens"] - cached - written
    if min(uncached, written - long_write) < 0:
        raise ValueError("inconsistent cached and uncached usage")
    return (
        uncached * rate_in + cached * rate_cached + row["output_tokens"] * rate_out
        + (written - long_write) * rate_in * 1.25 + long_write * rate_in * 2.5
    ) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="+", type=Path)
    args = parser.parse_args()
    seen = set()
    groups: dict[str, dict] = defaultdict(
        lambda: dict(calls=0, input_tokens=0, output_tokens=0, reasoning_tokens=0, usd=0.0),
    )
    for path in args.ledger:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = row["request_id"]
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            total = groups[row["operation"]]
            total["calls"] += 1
            total["input_tokens"] += row["input_tokens"]
            total["output_tokens"] += row["output_tokens"]
            total["reasoning_tokens"] += row["output_details"].get("reasoning_tokens", 0)
            total["usd"] += cost(row)
    result = {"cost_usd": round(sum(g["usd"] for g in groups.values()), 6),
              "operations": {k: {**v, "usd": round(v["usd"], 6)}
                             for k, v in sorted(groups.items())}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
