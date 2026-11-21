"""Deterministic provenance and consistency checks (§7.2 items 1-4).

1. Evidence chunk_id must be in the FINAL (post-budget) allowed context.
2. (Scope is guaranteed by retrieval: context comes only from the scoped query.)
3. The quote must match the chunk text under a documented, character-preserving
   matching rule (QUOTE_MATCH_VERSION): first strict (runs of whitespace -> one
   space), then whitespace-insensitive (PDF text layers split words: "уст ановленной"),
   and an ellipsis ("..." / "…") may stand for a skipped run of at most
   MAX_ELLIPSIS_GAP_CHARS inside the SAME chunk, segments in order, each segment
   >= MIN_SEGMENT_CHARS non-space characters. Every non-space character of the
   quote must appear verbatim and in order; nothing is fuzzy. The citation text,
   offsets and pages are ALWAYS the raw canonical span recovered server-side.
4. Numbers, percents, amounts and the working/calendar-days distinction in the
   claim text must be present in its evidence quotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.rules import COMMITMENT, conditional_rules
from app.ingestion.scopes import requested_scope_keys, scope_key
from app.qa.counts import unsupported_object_count
from app.qa.numbers import qualify_numeric_text, unsupported_numbers
from app.qa.schemas import Claim
from app.retrieval.clauses import local_list_lead
from app.retrieval.search import ContextChunk


@dataclass(frozen=True)
class LocatedQuote:
    chunk: ContextChunk
    quote_raw: str  # exact substring of the chunk text (and of canonical text)
    char_start: int  # canonical offsets
    char_end: int
    page: int
    page_to: int


class ProvenanceError(Exception):
    """A fabricated/foreign chunk, unlocatable quote, or inconsistent facts."""


class ConditionalWordingError(ProvenanceError):
    """A selected rule could be rendered directly without losing its conditions."""


def _ws_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


QUOTE_MATCH_VERSION = "ws-insensitive+ellipsis-v2"
MAX_ELLIPSIS_GAP_CHARS = 400
MIN_SEGMENT_CHARS = 12
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.+|…)\s*")


def _lenient_pattern(quote: str) -> str | None:
    """Whitespace-insensitive pattern; an ellipsis is a bounded in-order gap."""
    segments = [seg for seg in _ELLIPSIS.split(quote.strip()) if seg.strip()]
    if not segments:
        return None
    parts = []
    for seg in segments:
        chars = [ch for ch in seg if not ch.isspace()]
        if len(chars) < MIN_SEGMENT_CHARS and len(segments) > 1:
            return None  # tiny fragments must not be stitched together
        parts.append(r"\s*".join(re.escape(ch) for ch in chars))
    return f"[\\s\\S]{{0,{MAX_ELLIPSIS_GAP_CHARS}}}?".join(parts)


def locate_quote(chunk: ContextChunk, quote: str) -> LocatedQuote:
    """Find the quote in the chunk (strict, then the documented lenient rule);
    recover the RAW span and canonical offsets. Raises ProvenanceError if absent."""
    tokens = _ws_normalize(quote).split(" ")
    if not tokens or not tokens[0]:
        raise ProvenanceError("empty quote")
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, chunk.text)
    if match is None:
        lenient = _lenient_pattern(quote)
        match = re.search(lenient, chunk.text) if lenient else None
    if match is None:
        raise ProvenanceError(
            f"quote is not a substring of chunk {chunk.chunk_id} "
            "(after whitespace normalization)"
        )

    return quote_at_offsets(chunk, match.start(), match.end())


def quote_at_offsets(chunk: ContextChunk, local_start: int, local_end: int) -> LocatedQuote:
    """Keep the selected occurrence even if identical text appears twice in one chunk."""
    start = chunk.char_start + local_start
    end = chunk.char_start + local_end
    page, page_to = _pages_for_span(chunk, start, end)
    return LocatedQuote(
        chunk=chunk, quote_raw=chunk.text[local_start:local_end],
        char_start=start, char_end=end, page=page, page_to=page_to,
    )


def _pages_for_span(chunk: ContextChunk, start: int, end: int) -> tuple[int, int]:
    if chunk.segments:
        page = page_to = chunk.page_from
        for segment in chunk.segments:
            if segment["char_start"] <= start < segment["char_end"]:
                page = int(segment["page"])
            if segment["char_start"] < end <= segment["char_end"]:
                page_to = int(segment["page"])
        return page, max(page, page_to)
    return chunk.page_from, chunk.page_to


# Narrow fallback for an explicit request for a number answered by a cross-reference.
# The verifier handles units, conditional values, multipart and indirect questions.
_NUMERIC_QUESTION = re.compile(
    r"^\s*(?:какой процент|сколько процентов|какова сумма|каков размер|"
    r"сколько (?:рублей|дней|часов))\b", re.IGNORECASE
)
_REFERRED_VALUE = re.compile(
    r"согласован|соглашени|определя|установлен|указан|согласно|в соответствии|по тариф",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"\d|[%½¼¾]|\b(?:нол|нул|один|одн|дв[ауе]|три(?:\b|надц|дц)|тр[её]х|четыр|пят|шест|сем[ьи]|"
    r"вос[ье]|девят|десят|сорок|сто\b|сот|тысяч|миллион|половин|четверт|треть|"
    r"полност|в полном|бесплат|безвозмезд|без удержан|не (?:удерж|взима|начисл|требу))",
    re.IGNORECASE,
)


def check_requested_value(question: str, claims: list[Claim]) -> None:
    """An agreement or tariff reference alone does not supply a requested amount."""
    if _NUMERIC_QUESTION.search(question):
        answer = " ".join(c.text for c in claims)
        if _REFERRED_VALUE.search(answer) and not _QUANTITY.search(answer):
            raise ProvenanceError("вместо запрошенной величины приведён порядок её определения")


def check_claim_consistency(
    claim: Claim, quotes: list[str], question: str = "", names: tuple[str, ...] = (),
) -> None:
    """Check the role and unit of numeric facts before semantic verification."""
    joined = " ".join(quotes)
    missing = unsupported_numbers(claim.text, quotes, question, names)
    if missing:
        raise ProvenanceError(
            f"Значение или его роль не подтверждены: {', '.join(missing)}. "
            "Выбери фрагмент с названием, номером пункта или величиной с её единицей. "
            "Условие из вопроса можно опустить, указав применимое правило в единицах документа."
        )
    for kind in ("рабоч", "календарн"):
        if kind in claim.text.lower() and kind not in joined.lower():
            raise ProvenanceError(
                f"claim mentions '{kind}*' days but the evidence does not"
            )


_PERMISSION = re.compile(r"\b(?:вправе|име\w*\s+прав[оа]|разреш|"
                         r"запрещ|допуска[ею]|дозвол|мож(?:но|ет)|могут|нельзя)", re.I)
_ASKS_PERMISSION = re.compile(r"\b(?:вправе ли|разреш\w* ли|запрещ\w* ли|допускается ли|"
                              r"мож(?:но|ет) ли|могут ли|име\w* ли.{0,80}\bправ[оа]\b)", re.I)
_NUMBERED_ACTION = re.compile(r"^\s*\d+(?:\.\d+)+[.)]?\s+[а-яё]+(?:ть|ти|чь)(?:ся|сь)?\b", re.I)


def _check_item_permission(question: str, claim: Claim, quotes: list[LocatedQuote]) -> None:
    # An isolated numbered infinitive names an action. Without a governing
    # statement there is no evidence of who may do it or whether it is prohibited.
    if (_ASKS_PERMISSION.search(question) and _PERMISSION.search(claim.text)
            and all(_NUMBERED_ACTION.match(q.chunk.text) for q in quotes)
            and not any(_PERMISSION.search(q.quote_raw) for q in quotes)):
        raise ProvenanceError("перечисление действий без вводной фразы не устанавливает право")


_NEGATIVE = re.compile(r"\b(?:запрещ\w*|запрет(?:а|ом)?|нельзя|не\s+(?:вправе|"
                       r"допуска\w+|имеет права|может|разреш\w*))", re.I)
_AFFIRMATIVE = re.compile(r"\b(?:вправе|разреш\w*|допуска\w*|можно|имеет право)", re.I)


def _rule_text(value: str) -> str:
    # A PDF may split a word across lines; keep negation, conjunctions and order.
    return re.sub(r"\s+", "", value.casefold().replace("ё", "е")).strip(".«» ")


def _check_rule_meaning(question: str, claim: Claim, quotes: list[LocatedQuote]) -> None:
    text = claim.text
    evidence = " ".join(q.quote_raw for q in quotes)
    if "письменн" in text.lower() and "письменн" not in evidence.lower():
        raise ProvenanceError("источник не устанавливает письменную форму для этого действия")
    regulatory = _PERMISSION.search(text) or _ASKS_PERMISSION.search(question)
    for q in quotes:
        for left, right in conditional_rules(q.chunk.text):
            # A clipped excerpt can omit an exception at either end. Find the
            # complete rule containing it before validating a regulatory claim.
            if (q.char_end <= q.chunk.char_start + left
                    or q.char_start >= q.chunk.char_start + right):
                continue
            rule = q.chunk.text[left:right]
            if not regulatory and not COMMITMENT.search(rule):
                continue
            # A leading consent proviso can move intact to the end of a claim.
            # An exception to a prohibition needs the entire rule: repeating
            # its words under a wider permission changes their logical scope.
            consent = re.match(r"^(с (?:письменного )?согласия [^,.:;]{1,100}?)"
                               r"(?:,|\s+(?:допуска|разреш|вправе))", rule, re.I)
            required = consent[1] if consent and not _NEGATIVE.search(rule) else rule
            if COMMITMENT.search(rule):
                # For a conditional guarantee the answer may paraphrase the
                # promised outcome, but must preserve the attached proviso.
                tail = re.search(r",\s*((?:если|при условии(?:,? что)?)\b.+)", rule, re.I | re.S)
                if tail:
                    required = tail[1]
            if _rule_text(required) not in _rule_text(text):
                raise ConditionalWordingError(
                    "перенеси целиком правило с условиями и исключениями в отдельный claim: "
                    + " ".join(rule.split())
                )
    if not _PERMISSION.search(text):
        return
    positive_evidence = _AFFIRMATIVE.search(_NEGATIVE.sub("", evidence))
    if (_NEGATIVE.search(evidence) and not positive_evidence
            and not _NEGATIVE.search(text) and _AFFIRMATIVE.search(text)):
        raise ProvenanceError("источник запрещает указанное действие")
    if positive_evidence and not _NEGATIVE.search(evidence) and _NEGATIVE.search(text):
        raise ProvenanceError("цитата не устанавливает заявленный запрет")


def _check_scoped_claim(
    claim: Claim, quotes: list[LocatedQuote], requested: set[str], labels: set[str],
) -> set[str]:
    cited = {scope_key(q.chunk.scope_label) for q in quotes if q.chunk.scope_label}
    if requested and cited - requested:
        raise ProvenanceError("выбранная цитата относится к другой акции или приложению")
    if len(requested) > 1 and cited:
        named = requested_scope_keys(claim.text, labels)
        if len(cited) != 1 or named != cited:
            raise ProvenanceError("при сравнении укажи название и значение каждой области отдельно")
    return cited


def validate_claims(
    claims: list[Claim],
    context_by_id: dict[str, ContextChunk],
    *, question: str = "", spans: dict[str, LocatedQuote] | None = None,
    render_conditional_rules: bool = False,
) -> dict[int, list[LocatedQuote]]:
    """Deterministic §7.2 checks for every claim; returns located quotes per claim.

    Raises ProvenanceError on ANY fabricated chunk id, unlocatable quote or
    numeric/day-kind inconsistency — the caller decides between refusal and the
    single allowed correction.
    """
    located: dict[int, list[LocatedQuote]] = {}
    chunks = {chunk.chunk_id: chunk for chunk in context_by_id.values()}
    labels = {chunk.scope_label for chunk in chunks.values() if chunk.scope_label}
    requested = requested_scope_keys(question, labels)
    covered: set[str] = set()
    for index, claim in enumerate(claims):
        quotes: list[LocatedQuote] = []
        for evidence in claim.evidence:
            if isinstance(evidence, str):
                selected = (spans or {}).get(evidence.strip().strip("[]"))
                if selected is None or selected.chunk.chunk_id not in chunks:
                    raise ProvenanceError("неизвестный ID доказательства в доступных источниках")
                quote, chunk = selected, selected.chunk
            else:
                if spans is not None:
                    raise ProvenanceError("выбери ID фрагмента, отмеченный в источниках")
                # Previously saved drafts can still be checked against their original context.
                key = evidence.chunk_id.strip().strip("[]").strip()
                legacy_chunk = context_by_id.get(key)
                if legacy_chunk is None:
                    raise ProvenanceError("evidence source is not in the allowed context")
                chunk = legacy_chunk
                quote = locate_quote(chunk, evidence.quote)
            # Give the verifier and the reader the same governing sentence. The
            # item alone can name an action with no indication whether it is allowed.
            ancestor = chunk
            while ancestor.list_heading_id:
                parent = chunks.get(ancestor.list_heading_id)
                if parent is None:
                    raise ProvenanceError("в контексте отсутствует вводная фраза перечня")
                quotes.append(locate_quote(parent, parent.text))
                ancestor = parent
            if not chunk.list_heading_id:
                lead = local_list_lead(chunk.text, quote.char_start - chunk.char_start)
                if lead:
                    quotes.append(locate_quote(chunk, lead))
            quotes.append(quote)
        # A model can also explicitly cite the heading, or cite several of its
        # children. Use the authoritative span just once per claim.
        quotes = list({(q.chunk.chunk_id, q.char_start, q.char_end): q for q in quotes}.values())
        covered.update(_check_scoped_claim(claim, quotes, requested, labels))
        _check_item_permission(question, claim, quotes)
        try:
            _check_rule_meaning(question, claim, quotes)
        except ConditionalWordingError:
            # Only server-issued, already selected excerpts may supply the wording.
            # Keep separate selected consequences as well as the complete proviso.
            # Re-run the same check: a clipped excerpt still cannot supply the rule.
            if (not render_conditional_rules or spans is None
                    or sum(len(q.quote_raw) for q in quotes) > 1600):
                raise
            rendered = " ".join(_ws_normalize(q.quote_raw) for q in quotes)
            if len(requested) > 1 and quotes[0].chunk.scope_label:
                rendered = f"«{quotes[0].chunk.scope_label}». {rendered}"
            claim.text = rendered
            _check_scoped_claim(claim, quotes, requested, labels)
            _check_rule_meaning(question, claim, quotes)
        evidence_texts = [q.quote_raw for q in quotes]
        names = tuple(q.chunk.scope_label for q in quotes if q.chunk.scope_label)
        check_claim_consistency(claim, evidence_texts, question, names)
        if unsupported_object_count(question, claim.text, evidence_texts):
            raise ProvenanceError("цитаты указывают другую величину вместо количества объектов")
        located[index] = quotes
    if len(requested) > 1 and covered != requested:
        raise ProvenanceError("сравнение не охватывает все названные области")
    return located

def qualify_numeric_claims(claims: list[Claim], located: dict[int, list[LocatedQuote]]) -> None:
    for index, claim in enumerate(claims):
        try:
            claim.text = qualify_numeric_text(claim.text, [q.quote_raw for q in located[index]])
        except ValueError as exc:
            raise ProvenanceError(str(exc)) from exc
