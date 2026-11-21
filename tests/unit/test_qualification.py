"""The qualification gate counts omissions, unexpected additions and absent verdicts."""
from pathlib import Path

import pytest
from app.schemas.qa import NO_ANSWER_MESSAGE


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "eval"))
    import qualification_review

    return qualification_review


def test_qualification_corpus_and_unsigned_source_sheet(harness):
    cases = harness.qualification_cases()
    assert len(cases) == 60
    assert sum(c["entity_specific"] for c in cases) == 6
    source_sheet = harness.PACK / "source-review.md"
    assert harness.source_reviewer(source_sheet, {c["case_id"] for c in cases}) is None


def test_source_sheet_links_from_a_symlinked_directory(harness, tmp_path):
    import re

    from qualification_data import source_sheet

    folder = tmp_path / "files" / "worksheets"
    folder.mkdir(parents=True)
    alias = tmp_path / "current"
    alias.symlink_to(folder, target_is_directory=True)
    content = source_sheet(harness.qualification_cases(), alias / "review.md")
    links = re.findall(r"\]\(<([^>]+)>\)", content)
    assert links
    assert all((alias / link.split("#", 1)[0]).is_file() for link in links)


def test_extra_unlisted_claim_cannot_count_as_correct(harness):
    row = {
        "req_id": "a", "expected_found": True, "found": True, "required_facts": ["100 рублей"],
        "condition_count": 0, "claims": harness.segments("100 рублей. [1] И 9 месяцев."),
    }
    review = harness.blank_review(row)
    review["required_covered"] = [True]
    review["claims"][0].update(verdict="supported", critical=False)
    review["claims"][1].update(verdict="wrong_scope", critical=True, reason="другой объект")
    assert harness.check_review(row, review) == (1, 1, False, True, 1)
    review["claims"].pop()
    with pytest.raises(ValueError, match="unreviewed"):
        harness.check_review(row, review)


def test_missing_condition_and_false_refusal_lower_scores(harness):
    row = {
        "req_id": "b", "expected_found": True, "found": True,
        "required_facts": ["допускается", "только с согласия"], "condition_count": 1,
        "claims": ["Допускается. [1]"],
    }
    review = harness.blank_review(row)
    review["required_covered"] = [True, False]
    review["claims"][0].update(verdict="unsupported", critical=False, reason="пропущено условие")
    assert harness.check_review(row, review) == (1, 2, False, True, 1)
    row.update(found=False, claims=[], answer=NO_ANSWER_MESSAGE)
    assert harness.check_review(row, harness.blank_review(row)) == (0, 2, False, False, 0)


def test_frozen_real_packs_are_distinct_and_plan_every_probe(harness):
    import hashlib
    import json

    from qualification_data import pack_path, source_files

    documents = set()
    questions = set()
    for name in ("contracts-six", "published-six", "external-six"):
        pack = pack_path(name)
        locked = json.loads((pack / "lock.json").read_text())["files"]
        for path, expected in locked.items():
            assert hashlib.sha256((pack / path).read_bytes()).hexdigest() == expected
        pdfs = source_files(pack)
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs.values()}
        assert len(hashes) == 6
        assert hashes.isdisjoint(documents)
        documents.update(hashes)
        cases = [json.loads(line) for line in (pack / "gold.jsonl").read_text().splitlines()]
        unique = {c["question"].strip().casefold() for c in cases}
        assert len(unique) == len(cases) == 60
        assert unique.isdisjoint(questions)
        questions.update(unique)
        assert sum(not c["answerable"] for c in cases) == 20
        planned = harness.build_requests(cases, sorted(pdfs), "all")
        assert len({r["req_id"] for r in planned}) == len(planned) == 66
        for request in planned:
            if request["kind"] == "isolation":
                gold = next(c for c in cases if c["case_id"] == request["case_id"])
                assert request["target_fixture"] != gold["doc_fixture_id"]


def test_combined_catalog_includes_every_contract_and_scan_once(harness, tmp_path, monkeypatch):
    import json

    import catalog

    monkeypatch.setattr(catalog, "WORK", tmp_path)
    docs = catalog.prepare_gold()
    assert {f"{i:02}" for i in range(1, 16)} <= docs.keys()
    assert len(docs) == len({p.resolve() for p in docs.values()}) == 27
    assert all(p.is_relative_to(catalog.DOCUMENTS) for p in docs.values())
    assert sum(p.suffix == ".docx" for p in docs.values()) == 4
    rows = [json.loads(line) for line in (tmp_path / "gold.jsonl").read_text().splitlines()]
    assert len(rows) == 323
    assert sum(r["answerable"] for r in rows) == 220
    assert sum(r["doc_fixture_id"] == "06" for r in rows) == 10
    assert sum(r["doc_fixture_id"] == "04" for r in rows) == 19

    plan = json.loads((catalog.GOLD / "acceptance-plan.json").read_text())
    themed = plan["thematic_sets"].values()
    ids = plan["contract_cases"] + [qid for group in themed for qid in group["questions"]]
    by_id = {r["case_id"]: r for r in rows}
    assert len(ids) == len(set(ids)) == plan["questions"] == 210
    assert {by_id[qid]["doc_fixture_id"] for qid in ids} == set(docs)
    assert sum(by_id[qid]["answerable"] for qid in ids) == plan["expected_found"]
    assert sum(len(group["isolation"]) for group in themed) == plan["isolation_probes"] == 18
