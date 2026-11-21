"""Aggregate the four live series in the acceptance plan by question and document."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from paths import ROOT

sys.path.insert(0, str(ROOT))

from app.schemas.qa import AnswerResponse
from catalog import GOLD, PACKS, cases, documents
from qualification_data import digest
from run_qa import build_requests, load_jsonl

REPORTS = ROOT / "eval/reports"
SERIES = {"published-six": "published-final", "contracts-six": "contracts",
          "external-six": "offers", "numbered": "catalog"}


def aggregate(reports: Path) -> dict:
    docs = documents()
    bank = cases(docs)
    document_hashes = {fid: digest(path) for fid, path in docs.items()}
    by_id = {r["case_id"]: r for r in bank}
    plan = json.loads((GOLD / "acceptance-plan.json").read_text())
    totals: Counter = Counter()
    per_doc = {fid: Counter(questions_in_bank=sum(r["doc_fixture_id"] == fid for r in bank))
               for fid in docs}
    seen = set()
    manifests = {}
    build = None
    for key, folder in SERIES.items():
        directory = reports / folder
        raw = directory / "responses.jsonl"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        metrics = json.loads((directory / "metrics-all.json").read_text())
        provenance = metrics["provenance"]
        if (manifest.get("result_type") != "live" or manifest["responses_sha256"] != digest(raw)
                or provenance["live_manifest_sha256"] != digest(manifest_path)
                or provenance["responses_sha256"] != manifest["responses_sha256"]):
            raise ValueError(f"unbound or modified results in {key}")
        ids = (plan["contract_cases"] if key == "numbered" else
               plan["thematic_sets"][key]["questions"] + plan["thematic_sets"][key]["isolation"])
        inputs = manifest["inputs"]
        identity = (inputs["api_source_sha256"], inputs["runtime"]["images"],
                    inputs["runtime"]["llm_digest"], inputs["runtime"]["settings"])
        if build and build != identity:
            raise ValueError(f"the serving stack changed in {key}")
        build = identity
        manifests[key] = digest(manifest_path)
        group = bank if key == "numbered" else load_jsonl(PACKS / key / "gold.jsonl")
        planned = {r["req_id"]: r for r in build_requests(
            group, sorted({r["doc_fixture_id"] for r in group}), "all") if r["req_id"] in ids}
        records = load_jsonl(raw)
        if ({r["req_id"] for r in records} != set(ids) or len(records) != len(ids)
                or planned.keys() != set(ids) or len(records) != manifest["response_count"]):
            raise ValueError(f"incomplete or repeated requests in {key}")
        if seen.intersection(planned):
            raise ValueError(f"questions double-counted in {key}")
        seen.update(planned)
        for rec in records:
            req = planned[rec["req_id"]]
            fixture = req["target_fixture"]
            doc = inputs["documents"][fixture]
            if (any(rec[k] != v for k, v in req.items()) or rec["doc_id"] != doc["doc_id"]
                    or doc["source_sha256"] != document_hashes[fixture]):
                raise ValueError(f"wrong question or document for {rec['req_id']}")
            positive = by_id[rec["case_id"]]["answerable"]
            kind = ("isolation" if req["kind"] == "isolation"
                    else "positive" if positive else "negative")
            counts = per_doc[fixture]
            counts[kind] += 1
            totals[kind] += 1
            if rec["http_status"] != 200:
                continue
            try:
                answer = AnswerResponse.model_validate(rec["body"], strict=True)
            except ValueError:
                continue
            if answer.found is (kind == "positive"):
                counts[f"{kind}_correct_decision"] += 1
                totals[f"{kind}_correct_decision"] += 1
        for field in ("structured_validity_200", "citation_span_validity",
                      "cross_doc_leakage_citations"):
            totals[field + "_num"] += metrics[field]["num"]
            totals[field + "_den"] += metrics[field]["den"]
    if (len(seen) != plan["questions"] + plan["isolation_probes"]
            or len(per_doc) != plan["documents"]
            or any(not counts["positive"] or not counts["negative"] for counts in per_doc.values())
            or totals["positive"] != plan["expected_found"]):
        raise ValueError("the recorded runs do not cover the prescribed documents and questions")
    return {"documents": len(docs), "questions_in_bank": len(bank),
            "questions_run": totals["positive"] + totals["negative"],
            "isolation_probes": totals["isolation"], "app_sha256": build[0],
            "plan_sha256": digest(GOLD / "acceptance-plan.json"), "manifests_sha256": manifests,
            "totals": dict(totals), "by_document": per_doc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.reports), ensure_ascii=False, indent=2))
