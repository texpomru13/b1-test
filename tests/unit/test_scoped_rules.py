"""Regression cases from the published audit and structural variations."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.ingestion.canonical import CanonicalDocument, build_canonical
from app.ingestion.chunker import ApproxTokenLength, Chunk, split_into_chunks
from app.ingestion.pdf_loader import ExtractedPage, extract_pdf
from app.ingestion.scopes import requested_scope_keys
from app.ingestion.sections import detect_sections
from app.jobs.indexer import _chunk_payload
from app.qa.evidence import issue_evidence
from app.qa.grounding import ProvenanceError, validate_claims
from app.qa.schemas import Claim, Evidence
from app.retrieval.search import ContextChunk, hybrid_search
from app.storage.qdrant_repo import point_id

from tests.unit.test_retrieval_units import SCOPE, FakeBm25, FakeLM, FakeQdrant, make_settings

ROOT = Path(__file__).resolve().parents[2]


def split(doc: CanonicalDocument) -> list[Chunk]:
    return split_into_chunks(doc, detect_sections(doc.text), chunk_tokens=450,
                             overlap_tokens=64, length_fn=ApproxTokenLength())


def pages(*texts: str) -> CanonicalDocument:
    return build_canonical([ExtractedPage(i, text, True) for i, text in enumerate(texts, 1)])


def points(chunks: list[Chunk], indices: list[int]) -> list[SimpleNamespace]:
    raw = [asdict(chunk) for chunk in chunks]
    return [SimpleNamespace(
        id=point_id("doc", "run", i),
        payload=_chunk_payload(raw[i], job={"doc_id": "doc"}, run_id="run",
                               fingerprint=SCOPE.index_fingerprint, page_count=chunks[-1].page_to,
                               chunk_count=len(chunks), created_at="", chunks=raw),
    ) for i in indices]


class RecordingReranker(FakeLM):
    async def rerank(self, request):
        self.scored = {d.id: d.text for d in request.documents}
        return await super().rerank(request)


async def retrieve(chunks: list[Chunk], indices: list[int], question: str, **limits):
    candidates = points(chunks, indices)
    model = RecordingReranker({p.id: -float(i) for i, p in enumerate(candidates)})
    result = await hybrid_search(
        question=question, scope=SCOPE, settings=make_settings(**limits),
        qdrant=FakeQdrant(candidates), local_models=model, bm25=FakeBm25(), length_fn=len,
    )
    return result.context, model.scored


def claim(text: str, source: ContextChunk, quote: str | None = None) -> Claim:
    evidence = Evidence(chunk_id=source.chunk_id, quote=quote or source.text)
    return Claim(text=text, evidence=[evidence])


@pytest.mark.parametrize("reverse", [False, True])
async def test_reordered_offers_with_repeated_clauses_and_global_terms(reverse):
    offers = ['Акция «Ветер»\n3.1. Срок составляет 11 дней.',
              'Акция «Ветер плюс»\n3.1. Срок составляет 17 дней.']
    if reverse:
        offers.reverse()
    doc = pages(*offers, "Общие условия для всех акций\n2. Цена включает доставку.")
    chunks = split(doc)
    facts = [c for c in chunks if c.text.startswith(("3.1.", "2."))]
    assert all(doc.text[slice(*c.scope_label_span)] == c.scope_label
               for c in facts if c.scope_label_span)
    question = 'Каковы срок и условия доставки по акции «Ветер плюс»?'
    selected, scored = await retrieve(chunks, [c.index for c in facts], question)
    assert {c.scope_label for c in selected} == {"Ветер плюс", None}
    assert all(c.scope_label + "\n" in scored[point_id("doc", "run", c.index)]
               for c in facts if c.scope_label)
    no_value, _ = await retrieve(chunks, [c.index for c in facts if c.scope_label == "Ветер"],
                                 question)
    assert no_value == []  # the only period available belongs to a neighbour

    question = 'Сравните акции «Ветер» и «Ветер плюс».'
    both, _ = await retrieve(chunks, [c.index for c in facts if c.scope_label], question)
    assert len({c.scope_id for c in both}) == 2
    sources = {c.chunk_id: c for c in both}
    labelled = [claim(f'«{c.scope_label}»: {c.text.partition(". ")[2]}', c) for c in both]
    assert len(validate_claims(labelled, sources, question=question)) == 2
    with pytest.raises(ProvenanceError, match="сравнен"):
        validate_claims(labelled[:1], sources, question=question)
    labelled[0].text = "Срок составляет 11 дней."
    with pytest.raises(ProvenanceError, match="укажи название"):
        validate_claims(labelled, sources, question=question)


async def test_nested_rule_scored_and_budgeted_with_exact_ancestors_outside_candidates():
    doc = pages("Приложение № 1\n3.1. Без согласия владельца запрещено:\n"
                "3.1.1. Передавать третьим лицам:\n3.1.1.1. Карты.",
                "3.1.1.2. Ключи.", "Приложение № 2\n3.1. Иные правила.")
    chunks = split(doc)
    child = next(c for c in chunks if "Ключи" in c.text)
    q = "Можно ли передавать ключи по приложению № 1?"
    assert child.parent_clause_id == doc.text.index("3.1.1. Передавать")
    selected, scored = await retrieve(chunks, [child.index], q)
    assert len(selected) == 3 and [c.page_from for c in selected] == [1, 1, 2]
    unit = scored[point_id("doc", "run", child.index)]
    assert all(c.text in unit for c in selected)
    assert selected[2].list_heading_id == selected[1].chunk_id
    assert selected[1].list_heading_id == selected[0].chunk_id
    assert not (await retrieve(chunks, [child.index], q, context_top_k=2))[0]
    assert not (await retrieve(chunks, [child.index], q, context_budget_tokens=50))[0]
    assert not chunks[-1].governing_spans


def test_scope_names_inside_other_names_and_unknown_target():
    names = {"Базовый", "Базовый плюс"}
    assert requested_scope_keys("как действует Базовый плюс?", names) == {"базовый плюс"}
    assert requested_scope_keys('срок акции «Другой»?', names) == {"другой"}


def audited(case_id: str):
    fid = case_id.split("-")[0]
    doc = build_canonical(extract_pdf(next((ROOT / "test_documents").glob(fid + "_*.pdf")),
                                      max_pages=100).pages)
    archive = ROOT / "eval/reports/submission/responses.jsonl"
    row = next(rec for line in archive.read_text().splitlines()
               if (rec := json.loads(line))["req_id"] == case_id)
    for ref in row["body"]["citations"]:
        assert doc.text[ref["char_start"]:ref["char_end"]] == ref["text"]
    return row, split(doc)


async def test_14_03_archived_periods_cannot_enter_camera_answer():
    row, chunks = audited("14-03")
    refs = row["body"]["citations"]
    indices = [c.index for c in chunks if any(
        c.char_start < ref["char_end"] and c.char_end > ref["char_start"] for ref in refs
    )]
    selected, scored = await retrieve(chunks, indices, row["question"])
    assert len(scored) > len(selected)
    assert {c.scope_label for c in selected} == {"ВИДЕОКАМЕРА УЛИЧНАЯ ПОВОРОТНАЯ 4G"}
    sources = {f"source-{i}": c for i, c in enumerate(selected, 1)}
    spans = issue_evidence(sources)
    claims = []
    for i, text in [(0, "При единовременной оплате камера стоит 6 600 рублей с НДС."),
                    (4, "Максимальный промопериод — 14 календарных дней с даты подключения "
                        "оборудования в личном кабинете или мобильном приложении.")]:
        evidence = [k for k, q in spans.items() if q.char_start < refs[i]["char_end"]
                    and q.char_end > refs[i]["char_start"]]
        claims.append(Claim(text=text, evidence=evidence))
    verified = validate_claims(claims, sources, question=row["question"], spans=spans)
    assert {q.page for qs in verified.values() for q in qs} == {41, 42}
    assert not any("26 месяцев" in claim.text for claim in claims)


async def test_07_07_archive_rejected_and_complete_rule_with_sanction_is_supported():
    row, chunks = audited("07-07")
    rule, _, penalty = row["body"]["citations"]
    indices = [c.index for c in chunks if any(
        c.char_start <= ref["char_start"] and c.char_end >= ref["char_end"]
        for ref in [rule, penalty])]
    selected, _ = await retrieve(chunks, indices, row["question"])
    sources = {c.chunk_id: c for c in selected}
    rule_source = next(c for c in selected if rule["text"] in c.text)
    penalty_source = next(c for c in selected if penalty["text"] in c.text)
    bad = row["body"]["answer"].partition(" [1]")[0]
    with pytest.raises(ProvenanceError, match="письменную форму"):
        validate_claims([claim(bad, rule_source, rule["text"])], sources, question=row["question"])
    corrected = [claim(" ".join(rule["text"].split()), rule_source, rule["text"]),
                 claim(" ".join(penalty["text"].split()), penalty_source, penalty["text"])]
    verified = validate_claims(corrected, sources, question=row["question"])
    assert all(q.page == 9 for qs in verified.values() for q in qs)


@pytest.mark.parametrize("source,wrong", [
    ("Передача запрещена.", "Передача разрешена."),
    ("Передача разрешена.", "Передача запрещена."),
    ("Передача разрешена только при регистрации.", "Передача разрешена."),
    ("Копирование запрещено, кроме архивной копии.", "Все копии разрешены."),
    ("Доступ запрещён, кроме ремонта по согласованию с владельцем.",
     "Доступ разрешён по согласованию с владельцем."),
])
def test_modal_rule_preserves_direction_and_all_conditions(source, wrong):
    chunk = ContextChunk("x", 0, source, None, 1, 1, 0, len(source), .8)
    sources = {"x": chunk}
    with pytest.raises(ProvenanceError):
        validate_claims([claim(wrong, chunk)], sources, question="Можно ли это сделать?")
    assert validate_claims([claim(source, chunk)], sources, question="Можно ли это сделать?")


def test_05_03_conditional_storage_guarantee_applies_only_to_operator_hardware():
    doc = build_canonical(extract_pdf(next((ROOT / "test_documents").glob("05_*.pdf")),
                                      max_pages=100).pages)
    raw = next(c for c in split(doc) if "если эти данные\nхранились" in c.text)
    source = ContextChunk("x", raw.index, raw.text, raw.section, raw.page_from, raw.page_to,
                          raw.char_start, raw.char_end, .8)
    sources = {"x": source}
    spans = issue_evidence(sources)
    condition_id = next(k for k, q in spans.items() if "если эти данные" in q.quote_raw)
    duration = "Исполнитель гарантирует сохранность данных 14 дней с даты приостановления услуг"
    question = "Как долго исполнитель хранит данные после приостановления услуг?"
    with pytest.raises(ProvenanceError, match="условиями"):
        validate_claims([Claim(text=duration, evidence=[condition_id])], sources,
                        question=question, spans=spans)
    qualified = duration + ", если эти данные хранились на оборудовании Исполнителя."
    assert validate_claims([Claim(text=qualified, evidence=[condition_id])], sources,
                           question=question, spans=spans)

    short = Claim(text=duration, evidence=[condition_id])
    located = validate_claims([short], sources, question=question, spans=spans,
                              render_conditional_rules=True)
    assert short.text == " ".join(located[0][0].quote_raw.split())
    assert "если эти данные хранились на оборудовании Исполнителя" in short.text
    assert "автоматическом" not in short.text


def test_only_complete_selected_rule_can_supply_conditional_wording():
    text = "Доступ запрещён, кроме ремонта по согласованию с владельцем."
    chunk = ContextChunk("x", 0, text, None, 1, 1, 0, len(text), .8)
    sources = {"x": chunk}
    spans = issue_evidence(sources)
    claim = Claim(text="Доступ разрешён после согласования.", evidence=["x.1"])
    located = validate_claims([claim], sources, question="Можно ли получить доступ?",
                              spans=spans, render_conditional_rules=True)
    assert claim.text == text == located[0][0].quote_raw
    from app.qa.grounding import quote_at_offsets

    clipped = {"x.1": quote_at_offsets(chunk, 0, text.index(","))}
    with pytest.raises(ProvenanceError, match="условиями"):
        validate_claims([Claim(text="Можно получить доступ.", evidence=["x.1"])], sources,
                        question="Можно ли получить доступ?", spans=clipped,
                        render_conditional_rules=True)
