"""Verify one frozen group, prepare an exhaustive review and score its saved responses offline."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from completeness import DEST, ROOT, SERIES, load, plan, sha, unique
from qualification_data import write_once
from qualification_review import blank_review, check_review, segments


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def raw_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def frozen_selection(study: Path = DEST) -> dict:
    fixed = load(study / "plan.json")
    if study.name == "resolution":
        previous = load(DEST / "plan.json")
        original = unique(previous["requests"], "req_id")
        reviewed = {}
        files = [DEST / "plan.json", DEST / "result.json"]
        for series in SERIES:
            score_file = DEST / "series" / series / "result.json"
            group = load(score_file)
            if fixed["baseline_app_commit"] != group["app_commit"]:
                raise ValueError("previous answers came from a different application build")
            files.append(score_file)
            for r in group["requests"]:
                if r["expected_found"]:
                    reviewed[r["req_id"]] = r
        hashes = {str(f.relative_to(ROOT)): sha(f) for f in files}
        passed = sorted(key for key, row in reviewed.items() if row["passed"])
        failed = sorted(set(reviewed) - set(passed))
        controls = random.Random(int(fixed["seed_hex"], 16)).sample(
            passed, math.ceil(0.30 * len(passed))
        )
        rows = [
            dict(original[key], stratum=stratum, baseline_verdict=verdict,
                 baseline_found=reviewed[key]["found"])
            for keys, stratum, verdict in ((failed, "incomplete", "incomplete"),
                                          (controls, "complete_control", "complete"))
            for key in keys
        ]
        if (fixed["baseline_files_sha256"] != hashes or fixed["sorted_complete_ids"] != passed
                or fixed["requests"] != rows
                or fixed["expected_answerable_population"] != len(reviewed)):
            raise ValueError("resolution sample differs from its baseline or frozen draw")
        return fixed
    if plan((DEST / "seed.txt").read_text().strip()) != fixed:
        raise ValueError("selection or baseline differs from the committed plan")
    return fixed


def packet(series: str, work: Path, run: str,
           study: Path = DEST) -> tuple[dict, list[dict], list[str]]:
    selected = [r for r in frozen_selection(study)["requests"] if r["series"] == series]
    saved = work / "runs" / run
    manifest = load(saved / "manifest.json")
    records = unique(raw_rows(saved / "responses.jsonl"), "req_id")
    baseline_folder = ROOT / "eval/reports" / SERIES[series][0]
    if study.name == "resolution":
        baseline_folder = DEST / "series" / series
        old_labels = unique(load(baseline_folder / "result.json")["requests"], "req_id")
    else:
        previous_review = load(baseline_folder / SERIES[series][1])
        if series == "numbered":
            old_labels = unique(previous_review["cases"], "case_id")
        else:
            old_labels = unique(previous_review["rows"], "req_id")
    previous = load(baseline_folder / "manifest.json")
    manifest_hash = sha(saved / "manifest.json")
    gold = unique(raw_rows(work / "gold.jsonl"), "case_id")
    docs = load(work / "corpus.json")["docs"]
    if (
        manifest["result_type"] != "live"
        or manifest["run"] != run
        or manifest["response_count"] != len(selected)
        or set(records) != {r["req_id"] for r in selected}
        or manifest["responses_sha256"] != sha(saved / "responses.jsonl")
        or manifest["inputs"]["gold_sha256"] != sha(work / "gold.jsonl")
        or manifest["inputs"]["gold_sha256"] != previous["inputs"]["gold_sha256"]
        or manifest["inputs"]["documents"] != previous["inputs"]["documents"]
    ):
        raise ValueError(f"{series}: series, responses or rubric do not match the frozen plan")
    indices = {}
    members = ["corpus.json", "gold.jsonl"]
    for fid, item in manifest["inputs"]["documents"].items():
        name = f"index/{fid}.json"
        index = load(work / name)
        source = (ROOT / docs[fid]["file"]).resolve()
        if (
            not source.is_relative_to(ROOT)
            or sha(work / name) != item["export_sha256"]
            or index["doc_id"] != item["doc_id"]
            or docs[fid]["doc_id"] != index["doc_id"]
            or index["accepted_run_id"] != item["accepted_run_id"]
            or index["index_fingerprint"] != item["index_fingerprint"]
            or sha(source) != item["source_sha256"]
        ):
            raise ValueError(f"{series}/{fid}: source, document or accepted index changed")
        indices[fid] = index
        members.append(name)
    # This existing scorer verifies each returned citation against its target chunk, page and text.
    subprocess.run(
        [sys.executable, ROOT / "eval/metrics.py", "--run", run],
        cwd=ROOT,
        env={**os.environ, "B1_EVAL_WORK_DIR": str(work)},
        check=True,
        stdout=subprocess.DEVNULL,
    )
    checks = unique(raw_rows(saved / "cases-all.jsonl"), "req_id")
    if set(checks) != set(records):
        raise ValueError("technical checks are incomplete")
    rows = []
    for request in selected:
        rid = request["req_id"]
        fid = request["target_fixture"]
        rec, case, index = records[rid], gold[rid], indices[fid]
        if (
            rec["kind"] != "gold"
            or rec["case_id"] != rid
            or rec["target_fixture"] != fid
            or rec["doc_id"] != index["doc_id"]
            or rec["question"] != request["question"]
            or case["doc_fixture_id"] != fid
            or case["question"] != request["question"]
            or case["answerable"] != request["expected_found"]
        ):
            raise ValueError(f"{rid}: unexpected question, target or rubric")
        checked = checks[rid]
        body = rec["body"] if rec["http_status"] == 200 and checked.get("schema_valid") else {}
        citations = body.get("citations", [])
        spans = case.get("evidence", case["gold_spans"])
        text = index["canonical_text"]
        for span in spans:
            for match in span["occurrences"]:
                start, end = match["char_start"], match["char_end"]
                page = index["pages"][match["page"] - 1]
                if (
                    text[start:end] != span["text"]
                    or not page["char_start"] <= start < page["char_end"]
                ):
                    raise ValueError(f"{rid}: gold evidence no longer matches its page and offsets")
        conditions = case.get("required_conditions", [])
        if "required_facts" in case:
            facts = [fact["text"] for fact in case["required_facts"] + conditions]
        else:
            facts = case["expected_facts"] if case["answerable"] else []
        rows.append(
            {
                **request,
                "measurement_sha256": manifest_hash,
                "previous_assessment": old_labels[rid].get(
                    "assessment", old_labels[rid].get("coverage_note", "")
                ),
                "found": body.get("found"),
                "answer": body.get("answer", ""),
                "citations": citations,
                "claims": segments(body["answer"]) if body.get("found") else [],
                "required_facts": facts,
                "condition_count": len(conditions),
                "optional_facts": case.get("optional_facts", []),
                "known_errors": case["forbidden_facts"],
                "scope": case.get("scope", case.get("section")),
                "reference_answer": case["reference_answer"],
                "evidence_spans": spans,
                "source_file": docs[fid]["file"],
                "source_text": text,
                "schema_valid": bool(checked.get("schema_valid")),
                "citation_problems": checked.get("citation_problems", []),
            }
        )
    return manifest, rows, members


def assess(rows: list[dict], review: dict) -> list[dict]:
    if (
        review.get("reviewer_type") not in ("model", "human")
        or not review.get("reviewer", "").strip()
    ):
        raise ValueError("specify the actual reviewer name and type in claim-review.json")
    labels = unique(review["rows"], "req_id")
    if set(labels) != {r["req_id"] for r in rows}:
        raise ValueError("missing, duplicate or unplanned review decisions")
    results = []
    for row in rows:
        label = labels[row["req_id"]]
        hits, total, complete, critical, bad = check_review(row, label)
        complete = complete and not row["citation_problems"]
        if row["expected_found"]:
            success = complete
        else:
            success = row["found"] is False
        if not success and not str(label.get("assessment", "")).strip():
            raise ValueError(f"{row['req_id']}: explain the omission, wrong answer or refusal")
        results.append(
            {
                k: row[k]
                for k in (
                    "req_id",
                    "stratum",
                    "expected_found",
                    "found",
                    "baseline_verdict",
                    "schema_valid",
                    "citation_problems",
                )
            }
            | {
                "passed": bool(success),
                "covered": hits,
                "required": total,
                "bad_segments": bad,
                "critical": critical,
                "assessment": label.get("assessment", ""),
            }
        )
    return results


def totals(results: list[dict]) -> dict:
    strata = {}
    for stratum in ("incomplete", "complete_control", "false_positive"):
        cases = [r for r in results if r["stratum"] == stratum]
        strata[stratum] = {
            "passed": sum(r["passed"] for r in cases),
            "total": len(cases),
            "failed_ids": [r["req_id"] for r in cases if not r["passed"]],
        }
    return {
        "strata": strata,
        "critical_ids": [r["req_id"] for r in results if r["critical"]],
        "coverage": [sum(r["covered"] for r in results), sum(r["required"] for r in results)],
        "invalid_schema_ids": [r["req_id"] for r in results if not r["schema_valid"]],
        "invalid_citation_ids": [r["req_id"] for r in results if r["citation_problems"]],
    }


def archive_inputs(work: Path, names: list[str]) -> bytes:
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0) as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name in sorted(names):
                data = (work / name).read_bytes()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return compressed.getvalue()


def score(series: str, work: Path, run: str, out: Path, archive: Path | None,
          study: Path = DEST) -> dict:
    manifest, rows, names = packet(series, work, run, study)
    labels = load(out / "claim-review.json")
    results = assess(rows, labels)
    saved = work / "runs" / run
    result = {
        "series": series,
        "plan_sha256": sha(study / "plan.json"),
        "app_commit": manifest["git_commit"],
        "app_source_sha256": manifest["inputs"]["api_source_sha256"],
        "manifest_sha256": sha(saved / "manifest.json"),
        "responses_sha256": manifest["responses_sha256"],
        "review_sha256": sha(out / "claim-review.json"),
        "reviewer": labels["reviewer"],
        "reviewer_type": labels["reviewer_type"],
        **totals(results),
        "requests": results,
    }
    write_once(out / "result.json", encoded(result))
    if archive:
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "responses.jsonl"):
            write_once(archive / name, (saved / name).read_bytes())
        for name in ("claim-review.json", "result.json"):
            write_once(archive / name, (out / name).read_bytes())
        write_once(archive / "inputs.tar.gz", archive_inputs(work, names))
        files = [
            "manifest.json",
            "responses.jsonl",
            "claim-review.json",
            "result.json",
            "inputs.tar.gz",
        ]
        write_once(archive / "sha256.json", encoded({f: sha(archive / f) for f in files}))
    return result


def restore(series: str, folder: Path, work: Path) -> tuple[str, Path]:
    file_hashes = load(folder / "sha256.json")
    expected = {
        "manifest.json",
        "responses.jsonl",
        "claim-review.json",
        "result.json",
        "inputs.tar.gz",
    }
    if set(file_hashes) != expected or any(sha(folder / k) != v for k, v in file_hashes.items()):
        raise ValueError(f"{series}: archive inventory differs")
    manifest = load(folder / "manifest.json")
    docs = manifest["inputs"]["documents"]
    names = {"corpus.json", "gold.jsonl", *(f"index/{fid}.json" for fid in docs)}
    with tarfile.open(folder / "inputs.tar.gz") as archive:
        if (
            set(archive.getnames()) != names
            or len(archive.getmembers()) != len(names)
            or any(not member.isfile() for member in archive.getmembers())
        ):
            raise ValueError(f"{series}: unexpected input archive")
        archive.extractall(work, filter="data")
    run = manifest["run"]
    if Path(run).name != run or run in ("", ".", ".."):
        raise ValueError("invalid run name in manifest")
    out = work / "runs" / run
    out.mkdir(parents=True)
    for k in ("manifest.json", "responses.jsonl", "claim-review.json"):
        write_once(out / k, (folder / k).read_bytes())
    return run, out


def retained_population_interval(population: int, sample: int, retained: int) -> tuple[int, int]:
    """Invert the two equal-tailed hypergeometric tests, without floating-point thresholds."""
    if not 0 <= retained <= sample <= population or sample == 0:
        raise ValueError("unexpected control population, sample size or retained number")
    denominator = math.comb(population, sample)
    accepted = []
    for successes in range(population + 1):
        first = max(0, sample - population + successes)
        last = min(sample, successes)
        if not first <= retained <= last:
            continue
        left = right = 0
        for count in range(first, last + 1):
            mass = math.comb(successes, count) * math.comb(population - successes, sample - count)
            if count <= retained:
                left += mass
            if count >= retained:
                right += mass
        # Each tail must have probability >= 0.025 (= 1/40).
        if min(left, right) * 40 >= denominator:
            accepted.append(successes)
    return min(accepted), max(accepted)


def paired_and_estimated(strata: dict, population: int) -> dict:
    incomplete, controls = strata["incomplete"], strata["complete_control"]
    fixed, retained = incomplete["passed"], controls["passed"]
    before, after = controls["total"], fixed + retained
    observed = incomplete["total"] + before
    original_complete = population - incomplete["total"]
    estimate = fixed + original_complete * retained / before
    low, high = retained_population_interval(original_complete, before, retained)
    return {
        "observed_answerable_pairs": {
            "n": observed,
            "full_before": before,
            "full_after": after,
            "delta_percentage_points": 100 * (after - before) / observed,
            "incomplete_to_full": fixed,
            "full_to_incomplete": before - retained,
            "full_to_full": retained,
            "incomplete_to_incomplete": incomplete["total"] - fixed,
        },
        "estimated_population_after": {
            "n": population,
            "unmeasured": population - observed,
            "baseline_full": original_complete,
            "estimated_full": estimate,
            "estimated_full_percent": 100 * estimate / population,
            "retained_in_baseline_full_exact_95_percent_interval": [low, high],
            "estimated_full_exact_95_percent_interval": [fixed + low, fixed + high],
            "estimated_full_percent_exact_95_percent_interval": [
                100 * (fixed + low) / population,
                100 * (fixed + high) / population,
            ],
        },
    }


def summarize(root: Path, study: Path = DEST) -> dict:
    fixed = frozen_selection(study)
    combined, builds = [], set()
    for series in SERIES:
        directory = root / series
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp).resolve()
            run, out = restore(series, directory, work)
            result = score(series, work, run, out, None, study)
            if (out / "result.json").read_bytes() != (directory / "result.json").read_bytes():
                raise ValueError(f"{series}: scores differ on offline replay")
        combined.extend(result["requests"])
        runtime = load(directory / "manifest.json")
        builds.add(
            (
                result["app_commit"],
                result["app_source_sha256"],
                runtime["api"],
                json.dumps(runtime["inputs"]["runtime"], sort_keys=True),
            )
        )
    if len(builds) != 1 or set(unique(combined, "req_id")) != {
        r["req_id"] for r in fixed["requests"]
    }:
        raise ValueError("a group, question or common serving build is missing")
    aggregate = totals(combined)
    return {
        "plan_sha256": sha(study / "plan.json"),
        "app_commit": next(iter(builds))[0],
        "requests": len(combined),
        **aggregate,
        **paired_and_estimated(aggregate["strata"], fixed["expected_answerable_population"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["completeness", "resolution"], default="completeness")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "score"):
        command = sub.add_parser(name)
        command.add_argument("--series", choices=tuple(SERIES), required=True)
        command.add_argument("--work", type=Path, required=True)
        command.add_argument("--run", required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "score":
            command.add_argument("--archive", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", type=Path)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = DEST.parent / args.study
    if args.action == "summarize":
        result = summarize(args.root or study / "series", study)
        if args.output:
            write_once(args.output, encoded(result))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    work, out = args.work.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        _, rows, _ = packet(args.series, work, args.run, study)
        write_once(out / "review-input.json", encoded(rows))
        target = out / "claim-review.json"
        if not target.exists():
            write_once(
                target,
                encoded(
                    {
                        "reviewer": "",
                        "reviewer_type": None,
                        "rows": [dict(blank_review(r), assessment="") for r in rows],
                    }
                ),
            )
        elif [r["input_sha256"] for r in load(target)["rows"]] != [
            blank_review(r)["input_sha256"] for r in rows
        ]:
            raise ValueError("an existing review belongs to different inputs")
        print(len(rows), "requests prepared")
    else:
        result = score(args.series, work, args.run, out, args.archive, study)
        print(json.dumps(result["strata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
