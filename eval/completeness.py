"""Freeze or verify a retrievable, one-draw sample from the recorded acceptance runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval" / "reports"
DEST = REPORTS / "completeness"
SERIES = {
    "numbered": ("catalog", "numbered-review.json", None),
    "published-six": ("published-final", "claim-review.json", "qualification.json"),
    "contracts-six": ("contracts", "claim-review.json", "qualification.json"),
    "external-six": ("offers", "crosscheck-claims.json", "crosscheck.json"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows: list[dict], key: str) -> dict[str, dict]:
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key}")
    return result


def baseline(series: str) -> tuple[dict, list[dict]]:
    directory, review_name, summary_name = SERIES[series]
    folder = REPORTS / directory
    manifest = load(folder / "manifest.json")
    metrics = load(folder / "metrics-all.json")
    review = load(folder / review_name)
    raw_path = folder / "responses.jsonl"
    raw = unique([json.loads(s) for s in raw_path.read_text().splitlines()], "req_id")
    if (
        manifest["result_type"] != "live"
        or manifest["response_count"] != len(raw)
        or manifest["responses_sha256"] != sha(raw_path)
        or metrics["provenance"]["live_manifest_sha256"] != sha(folder / "manifest.json")
    ):
        raise ValueError(f"{series}: inconsistent baseline manifest")
    with tarfile.open(folder / "inputs.tar.gz") as archive:
        encoded = archive.extractfile("gold.jsonl").read()
    if hashlib.sha256(encoded).hexdigest() != manifest["inputs"]["gold_sha256"]:
        raise ValueError(f"{series}: rubric differs from manifest")
    gold = unique([json.loads(line) for line in encoded.splitlines()], "case_id")
    files = ["manifest.json", "metrics-all.json", "responses.jsonl", "inputs.tar.gz", review_name]
    if summary_name:
        files.append(summary_name)
        summary = load(folder / summary_name)
        if (
            summary["review_sha256"] != sha(folder / review_name)
            or summary["responses_sha256"] != sha(raw_path)
            or summary["metrics_sha256"] != sha(folder / "metrics-all.json")
        ):
            raise ValueError(f"{series}: semantic scores differ from raw results")
        labels = unique(review["rows"], "req_id")
    else:
        if review["responses_sha256"] != sha(raw_path):
            raise ValueError("numbered review differs from raw results")
        labels = unique(review["cases"], "case_id")
    if set(labels) != set(raw):
        raise ValueError(f"{series}: missing or extra assessments")
    rows = []
    for req_id, rec in raw.items():
        case, label = gold[rec["case_id"]], labels[req_id]
        if rec["kind"] != "gold":
            if rec["http_status"] != 200 or rec["body"]["found"] is not False:
                raise ValueError(f"baseline isolation failure needs separate enrollment: {req_id}")
            continue
        found = rec["body"]["found"] if rec["http_status"] == 200 else None
        if rec["question"] != case["question"] or rec["target_fixture"] != case["doc_fixture_id"]:
            raise ValueError(f"{req_id}: question or document differs from rubric")
        if summary_name:
            facts = case["required_facts"] + case["required_conditions"]
            hits = label["required_covered"]
            if len(hits) != len(facts) or any(type(v) is not bool for v in hits):
                raise ValueError(f"{req_id}: incomplete coverage labels")
            full = bool(
                found and all(hits) and all(c["verdict"] == "supported" for c in label["claims"])
            )
        else:
            if (
                label["question"] != rec["question"]
                or label["found"] != found
                or label["answer"] != rec["body"]["answer"]
            ):
                raise ValueError(f"{req_id}: numbered review and answer differ")
            full = label["verdict"] == "full"
        if case["answerable"]:
            status = "complete" if full else "incomplete"
        elif found is False:
            continue
        else:
            status = "false_positive" if found is True else "service_error"
        rows.append(
            {
                "req_id": req_id,
                "series": series,
                "target_fixture": rec["target_fixture"],
                "question": rec["question"],
                "expected_found": case["answerable"],
                "baseline_found": found,
                "baseline_verdict": status,
            }
        )
    hashes = {str((folder / name).relative_to(ROOT)): sha(folder / name) for name in files}
    return {
        "app_commit": manifest["git_commit"],
        "app_source_sha256": manifest["inputs"]["api_source_sha256"],
        "files_sha256": hashes,
    }, rows


def plan(seed: str) -> dict:
    if len(seed) != 64 or int(seed, 16) < 0:
        raise ValueError("seed must be a 256-bit hexadecimal value")
    sources, population = {}, []
    for series in SERIES:
        sources[series], rows = baseline(series)
        population.extend(rows)
    by_id = unique(population, "req_id")
    complete = sorted(k for k, r in by_id.items() if r["baseline_verdict"] == "complete")
    incomplete = sorted(k for k, r in by_id.items() if r["baseline_verdict"] == "incomplete")
    extra = sorted(k for k, r in by_id.items() if not r["expected_found"])
    if (len(complete), len(incomplete), extra) != (99, 36, ["contract-04-08"]):
        raise ValueError(
            "baseline differs from the specified 99 complete, 36 incomplete and one false positive"
        )
    sample = random.Random(int(seed, 16)).sample(complete, k=30)
    selected = [
        {**by_id[k], "stratum": stratum}
        for stratum, keys in (
            ("incomplete", incomplete),
            ("complete_control", sample),
            ("false_positive", extra),
        )
        for k in keys
    ]
    return {
        "schema_version": 1,
        "seed_hex": seed,
        "sampling": (
            "CPython 3.12 random.Random(int(seed_hex, 16)).sample(sorted_complete_ids, k=30)"
        ),
        "acceptance_plan_sha256": sha(ROOT / "eval/gold/acceptance-plan.json"),
        "baseline_sources": sources,
        "expected_answerable_population": 135,
        "sorted_complete_ids": complete,
        "counts": dict(Counter(r["stratum"] for r in selected)),
        "requests": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "check-plan"])
    parser.add_argument(
        "--out", type=Path, default=DEST, help="directory containing seed.txt and plan.json"
    )
    args = parser.parse_args()
    folder = args.out
    seed_file, destination = folder / "seed.txt", folder / "plan.json"
    if args.action == "freeze":
        folder.mkdir(parents=True, exist_ok=True)
        if not seed_file.exists():
            with seed_file.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_hex(32) + "\n")
    result = plan(seed_file.read_text().strip())
    data = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if args.action == "check-plan" or destination.exists():
        if destination.read_bytes() != data:
            raise ValueError("saved plan differs from the baseline, draw or case definitions")
    else:
        with destination.open("xb") as stream:
            stream.write(data)
    print(
        sha(destination),
        result["counts"],
        "groups:",
        dict(Counter(r["series"] for r in result["requests"])),
    )


if __name__ == "__main__":
    main()
