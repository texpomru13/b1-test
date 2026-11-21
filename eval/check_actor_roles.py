"""Exercise the production verifier against active, passive and list constructions."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
from paths import ROOT

sys.path.insert(0, str(ROOT))
from app.config.settings import Settings
from app.providers.factory import build_chat_adapter
from app.qa.prompts import VERIFIER_SYSTEM_V1, verifier_user_prompt
from app.qa.roles import reversed_list_actor
from app.qa.schemas import VERIFY_JSON_SCHEMA, Verification


def fixtures() -> list[tuple[str, str, list[tuple[str, list[str], bool]]]]:
    question = "Кто передаёт реестр заявок методическому совету?"
    active = ("Технический комитет:\n(i) ведёт учёт поступивших заявок;\n"
              "(ii) передаёт методическому совету реестр заявок.")
    passive = "Реестр заявок передаётся методическому совету техническим комитетом."
    direct = "Технический комитет передаёт реестр заявок методическому совету."
    correct_passive = "Реестр заявок передаётся методическому совету техническим комитетом."
    reversed_active = "Методический совет передаёт реестр заявок техническому комитету."
    reversed_passive = "Реестр заявок передаётся техническому комитету методическим советом."
    cases = [
        ("active_list", question, [(direct, [active], True), (reversed_active, [active], False),
                                   (correct_passive, [active], True)]),
        ("passive", question, [(reversed_passive, [passive], False), (direct, [passive], True)]),
    ]
    path = ROOT / "eval/reports/published-final/responses.jsonl"
    case = next(r for line in path.read_text().splitlines()
                if (r := json.loads(line))["req_id"] == "nice-06")
    body = case["body"]
    texts = {str(c["id"]): c["text"] for c in body["citations"]}
    claims = []
    for phrase, markers in re.findall(r"([^\[\]]+?)\s*((?:\[\d+\])+)", body["answer"]):
        citations = [texts[key] for key in re.findall(r"\d+", markers)]
        claims.append((phrase.strip(), citations, True))
    assert len(claims) == 5
    claims[-1] = (*claims[-1][:2], False)
    cases.append(("supplementary_claim", case["question"], claims))
    return cases


async def run(url: str, output: Path) -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="qwen3.5:9b-q4_K_M",
                        ollama_base_url=url, llm_temperature=0, llm_thinking="off",
                        llm_reasoning_budget_tokens=0, local_context_tokens=16384)
    adapter = build_chat_adapter(settings)
    async with httpx.AsyncClient() as client:
        response = await client.get(url.rstrip("/") + "/api/tags")
        response.raise_for_status()
        digest = next(m["digest"] for m in response.json()["models"]
                      if m["name"] == settings.llm_model)
    failures = []
    with output.open("x", encoding="utf-8") as out:
        for case_id, question, claims in fixtures():
            blocks = []
            for index, (claim, citations, _expected) in enumerate(claims):
                quotes = "\n".join(f'  ЦИТАТА (раздел не указан): "{text}"' for text in citations)
                blocks.append(f"CLAIM {index}: {claim}\n{quotes}")
            prompt = verifier_user_prompt(question, "\n\n".join(blocks))
            raw, usage = await adapter.generate_json(system=VERIFIER_SYSTEM_V1, user=prompt,
                                                     json_schema=VERIFY_JSON_SCHEMA, max_tokens=512)
            verification = Verification.model_validate(raw)
            verdicts = {v.claim_index: v.verdict.value for v in verification.verdicts}
            expected = [c[2] for c in claims]
            rejected_leads = [reversed_list_actor(claim, citations)
                              for claim, citations, _ in claims]
            effective = [verdicts.get(i) == "supported" and lead is None
                         for i, lead in enumerate(rejected_leads)]
            passed = set(verdicts) == set(range(len(claims))) and effective == expected
            entry = {"case": case_id, "model": settings.llm_model, "model_digest": digest,
                     "system_sha256": hashlib.sha256(VERIFIER_SYSTEM_V1.encode()).hexdigest(),
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "expected_supported": expected, "verifier": raw, "usage": usage,
                     "guard_rejected_leads": rejected_leads, "effective_supported": effective,
                     "passed": passed}
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()
            print(case_id, passed, flush=True)
            if not passed:
                failures.append(case_id)
    if failures:
        raise SystemExit("Unexpected verdicts: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.ollama_url, args.output))
