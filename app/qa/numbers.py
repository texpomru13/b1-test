"""Typed checks on numeric evidence before semantic verification."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.qa.grounding import LocatedQuote

_NUMBER = r"\d+(?:[.,]\d+)?"
_AMOUNT = re.compile(r"(?<!\d)\d{1,3}(?:[ \t\xa0\u202f]+\d{3})+(?!\d)")
_NAMED_ENTITY = re.compile(r'«[^«»\n]*\d[^«»\n]*»|"[^"\n]*\d[^"\n]*"')
_CLAUSE = re.compile(
    r"(?:\b(?i:п\.|пп\.|пункт\w*|подпункт\w*)\s*(?P<ref>\d+(?:\.\d+)+)"
    r"|(?m:^\s*)(?P<heading>\d+(?:\.\d+)+)(?:\.(?=\s)|(?=\s+[А-ЯЁ])))"
)
_NEXT_CLAUSE = re.compile(r"\s*\.?\s*(?:,|и|или)\s*(\d+(?:\.\d+)+)\b")
_DOCUMENT_ID_PATTERN = (
    r"\b(?P<kind>протокол|договор|приказ|приложени|положени|постановлени|соглашени)"
    r"(?:а|е|я|ю|у|ем|ом|ы|и|й|ей|ями|ям|ях|ов|ам|ами|ах)?[ \t]*(?:№|N(?:o|°)?\.?)[ \t]*"
    r"(?P<value>\d\w*(?:[-/.]\w+)*)(?![\w/])"
)
_DOCUMENT_ID = re.compile(_DOCUMENT_ID_PATTERN, re.I)
DOCUMENT_TITLE = re.compile(r"^[ \t]*" + _DOCUMENT_ID_PATTERN + r"[ \t]*$", re.M | re.I)
_DOC_LABEL = {"протокол": "протокола", "договор": "договора", "приказ": "приказа",
              "приложени": "приложения", "положени": "положения",
              "постановлени": "постановления", "соглашени": "соглашения"}
_ARTICLE = re.compile(r"\b(?:стать(?:я|е|ю|и|ёй|ей)|ст\.)\s*"
                      r"(?P<value>\d+(?:\.\d+)*)(?!\w)", re.I)
_SECTION_LINE = re.compile(r"^[ \t]*(\d+(?:\.\d+)*)(?:[.)][ \t]+|[ \t]+(?=[А-ЯЁ]))[^\n]+", re.M)
_PERCENT = r"%|п\s*р\s*о\s*ц\s*е\s*н\s*т\w*"
_UNITS = {
    "процент": _PERCENT,
    "рубль": r"руб(?:\.|л\w*)?\b|₽",
    "месяц": r"мес\.|месяц\w*\b",
    "год": r"(?:год\w*|лет)\b",
    "день": r"(?:рабоч\w*\s+|календарн\w*\s+)?(?:день|дня|дней|дни)\b",
    "час": r"час\w*\b",
    "минута": r"мин\.|минут\w*\b",
    "единица": r"(?:единиц\w*|штук\w*)\b",
}
_UNIT_PATTERN = re.compile(
    r"\s*(?:\([^()\d]{1,100}\)\s*)?(?:-х\s+)?(?:"
    + "|".join(f"(?P<u{i}>{pattern})" for i, pattern in enumerate(_UNITS.values())) + ")",
    re.I,
)
_REGISTRATION = re.compile(
    r"\b(ИНН|КПП|ОГРН)(?:\s*/\s*(ИНН|КПП))?\s*[:№]?\s*"
    r"(?:ООО\s+|АО\s+|ПАО\s+)?(?:составляет\s+)?(\d{6,})(?:\s*/\s*(\d{6,}))?\b", re.I,
)
_NUMBER_RE = re.compile(_NUMBER)
_PRICE_COLUMN = re.compile(
    r"\b(?:стоимость|цена)\s*[,(:]\s*(?:руб[.\-]?|рублей|₽)\)?(?:\s+с\s+НДС)?\s*$", re.I
)
_ROW_PRICE = re.compile(r"\s(\d+(?:[.,]\d+)?)\s*$")

# Only a full, explicit condition from the question may be repeated as a hypothetical.
_CONDITION = re.compile(r"\bесли\s+([^,;:.?!]+)", re.I)


_YEARS_IN_MONTHS = re.compile(
    r"\b(\d{1,5}|один|два|три|четыре) (?:год|года|лет)\s*\((\d{1,8}) месяц\w*\)", re.I
)


def _without_month_conversion(condition: str) -> str:
    def expand(match: re.Match[str]) -> str:
        words = {"один": 1, "два": 2, "три": 3, "четыре": 4}
        years = int(match[1]) if match[1].isascii() and match[1].isdigit() else words.get(match[1])
        if years is not None and 12 * years == int(match[2]):
            return match[0].split("(")[0].strip()
        return match[0]

    return _YEARS_IN_MONTHS.sub(expand, condition)


@dataclass(frozen=True)
class NumericFact:
    value: str
    role: str


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _amount_value(raw: str) -> str:
    whole, _, fraction = raw.replace(",", ".").partition(".")
    return ((whole.lstrip("0") or "0") + "." + fraction.rstrip("0")).rstrip(".")


def clause_reference_heading(claim: str, text: str, offset: int) -> tuple[int, int] | None:
    """Cite the actual numbered lead when the chosen span lies later in its list."""
    refs = {m["ref"] for m in _CLAUSE.finditer(claim) if m["ref"]}
    if not refs:
        return None
    current = None
    for heading in _SECTION_LINE.finditer(text):
        if heading.start() > offset:
            break
        current = heading
    if (current and current.end() <= offset and current[1] in refs
            and current[0].rstrip().endswith(":")):
        return current.span()
    return None


def numeric_evidence_texts(quotes: list[LocatedQuote]) -> list[str]:
    """Recover a terminal currency cell using its verified, separately cited header."""
    texts = [q.quote_raw for q in quotes]
    headers = {q.chunk.chunk_id: q for q in quotes if _PRICE_COLUMN.search(q.quote_raw)}
    rows = [(q, headers[parent_id]) for q in quotes
            if (parent_id := q.chunk.list_heading_id) is not None and parent_id in headers]
    # If a claim selects multiple rows, the verifier needs a draft identifying
    # each price with its own row. Never pool bare numeric cells across rows.
    if len(rows) == 1:
        row, header = rows[0]
        text = row.quote_raw.strip()
        if (header.char_end <= row.char_start and text == row.chunk.text.strip()
                and "\n" not in text and _ROW_PRICE.search(text)):
            texts.append(f"{header.quote_raw}\n{text}")
    return texts


def _facts(text: str) -> set[NumericFact]:
    # Replace classified ranges with spaces of the same length so other roles
    # cannot reuse e.g. the "1" in a company name as an item count.
    masked = list(text)
    facts: set[NumericFact] = set()
    for match in _NAMED_ENTITY.finditer(text):
        facts.add(NumericFact(_key(match[0][1:-1]), "название"))
        masked[match.start():match.end()] = " " * len(match[0])
    for match in _REGISTRATION.finditer("".join(masked)):
        facts.add(NumericFact(match[3], match[1].upper()))
        if match[2] and match[4]:
            facts.add(NumericFact(match[4], match[2].upper()))
        masked[match.start():match.end()] = " " * len(match[0])
    for pattern in (_DOCUMENT_ID, _ARTICLE):
        for match in pattern.finditer("".join(masked)):
            kind = "статья" if pattern is _ARTICLE else "номер " + _DOC_LABEL[match["kind"].lower()]
            facts.add(NumericFact(match["value"].lower(), kind))
            masked[match.start():match.end()] = " " * len(match[0])
    for match in _CLAUSE.finditer("".join(masked)):
        facts.add(NumericFact(match["ref"] or match["heading"], "пункт"))
        end = match.end()
        while continuation := _NEXT_CLAUSE.match(text, end):
            facts.add(NumericFact(continuation[1], "пункт"))
            end = continuation.end()
        masked[match.start():end] = " " * (end - match.start())
    cleaned = _AMOUNT.sub(lambda m: "".join(m.group().split()), "".join(masked))
    # In a text-layer table the price is often the rightmost cell under a
    # currency-bearing heading. Only carry that unit through contiguous data rows.
    price_column = False
    for line in cleaned.splitlines():
        if _PRICE_COLUMN.search(line):
            price_column = True
        elif price_column:
            if row := _ROW_PRICE.search(line):
                facts.add(NumericFact(_amount_value(row[1]), "рубль"))
            else:
                price_column = False
    for match in _NUMBER_RE.finditer(cleaned):
        unit = _UNIT_PATTERN.match(cleaned, match.end())
        role = list(_UNITS)[int(unit.lastgroup[1:])] if unit and unit.lastgroup else "число"
        raw = match[0].replace(",", ".")
        value = _amount_value(raw) if role != "число" else raw
        facts.add(NumericFact(value, role))
    return facts


def unsupported_numbers(
    claim: str, quotes: list[str], question: str, names: tuple[str, ...] = (),
) -> list[str]:
    evidence = {fact for quote in quotes for fact in _facts(quote)}
    question_conditions = {_key(m[1]) for m in _CONDITION.finditer(question)}
    # Do not treat arbitrary numbers present in the question as evidence. A
    # condition must be the same complete premise, marked as such in both texts.
    stated = _CONDITION.sub(
        lambda m: "" if _without_month_conversion(_key(m[1])) in question_conditions
        and claim[m.end():].lstrip().startswith(",") else m[0], claim
    )
    missing = []
    for fact in _facts(stated):
        if fact in evidence:
            continue
        if fact.role == "название" and any(
            re.search(r"(?<!\w)" + re.escape(fact.value) + r"(?!\w)", _key(text))
            for text in (*quotes, *names)
        ):
            # A table may print the same entity name without quotation marks.
            continue
        if fact.role == "число" and any(
            other.value == fact.value and other.role in {"ИНН", "КПП", "ОГРН", "единица"}
            for other in evidence
        ):
            # Support abbreviated cross-references and quantities using the object name.
            continue
        missing.append(f"{fact.value} ({fact.role})")
    return sorted(missing)

# A quantity can be an upper/lower bound, or a price under one payment method.
# Restore only qualifiers bound to this same value and unit in the cited passage.
_MEASURE_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[ \t\xa0\u202f]+\d{3})+|\d+)(?:[.,]\d+)?")
_LIMIT_WORD = re.compile(
    r"\b(?:(?P<upper>максимальн\w*)|(?P<lower>минимальн\w*))\s+"
    r"(?P<noun>продолжительност|длительност|срок|размер|сумм|стоимост|цен|количеств|ставк|"
    r"(?:промо[ -]?)?период)\w*",
    re.I,
)
_DIRECT_LIMIT = re.compile(r"\b(не\s+более|не\s+больше|не\s+менее|не\s+меньше|до)\s*$", re.I)
_PAYMENT = {
    "единовременн": "при единовременной оплате",
    "рассрочк": "при оплате в рассрочку",
}


def _measures(text: str) -> Iterator[tuple[NumericFact, int, int]]:
    for match in _MEASURE_NUMBER.finditer(text):
        unit = _UNIT_PATTERN.match(text, match.end())
        if unit and unit.lastgroup:
            role = list(_UNITS)[int(unit.lastgroup[1:])]
            value = _amount_value("".join(match[0].split()))
            yield NumericFact(value, role), match.start(), unit.end()


_BAND_AMOUNT = r"(?:\d+(?:[.,]\d+)?|один|одного|одну|одной)"
_BAND_UNIT = "(?:" + "|".join(_UNITS.values()) + ")"
_BAND = re.compile(
    rf"\b(?:(?:от|с)\s+(?P<a>{_BAND_AMOUNT})\s*(?P<ua>{_BAND_UNIT})?\s+"
    rf"(?:до|по)\s+(?P<b>{_BAND_AMOUNT})\s*(?P<ub>{_BAND_UNIT})"
    rf"|(?P<op>свыше|более|менее|до|от|не менее|не более)\s+"
    rf"(?P<n>{_BAND_AMOUNT})\s*(?P<u>{_BAND_UNIT}))", re.I,
)
_ROW_SEPARATOR = re.compile(r"\s*[—–:=|]\s*|\s+-\s+|[ ]*\t[ \t]*")


def _band_key(match: re.Match[str]) -> tuple[str, ...]:
    def endpoint(value: str, unit: str) -> tuple[str, str]:
        role = next(k for k, pattern in _UNITS.items() if re.fullmatch(pattern, unit, re.I))
        modifier = re.match(r"рабоч|календарн", unit, re.I)
        return (_amount_value(value) if value[0].isdigit() else "1",
                (modifier[0].lower() + " " if modifier else "") + role)

    if match["op"]:
        op = match["op"].lower()
        return ("свыше" if op == "более" else op, *endpoint(match["n"], match["u"]))
    return ("от-до", *endpoint(match["a"], match["ua"] or match["ub"]),
            *endpoint(match["b"], match["ub"]))


def _check_range_rows(text: str, quotes: list[str]) -> list[tuple[int, int]]:
    """Only the value in the row for the stated interval can support that pairing."""
    rows: dict[tuple[str, ...], set[NumericFact]] = {}
    for quote in quotes:
        for band in _BAND.finditer(quote):
            separator = _ROW_SEPARATOR.match(quote, band.end())
            if not separator:
                continue
            cell = quote[separator.end():separator.end() + 120]
            if value := next(_measures(cell), None):
                fact, start, _ = value
                if start == 0:
                    rows.setdefault(_band_key(band), set()).add(fact)
    if not rows:
        return []
    roles = {fact.role for facts in rows.values() for fact in facts}
    checked = []
    # One claim may enumerate multiple condition/value pairs in document order.
    # Reversed wording (value before condition) is unambiguous for a single pair.
    bands = list(_BAND.finditer(text))
    outputs = {fact for fact, start, _ in _measures(text)
               if fact.role in roles and not any(b.start() <= start < b.end() for b in bands)}
    if not bands and any(facts & outputs for facts in rows.values()) and len(rows) > 1:
        raise ValueError("Укажи диапазон, для которого приведено значение из таблицы.")
    for i, band in enumerate(bands):
        right = bands[i + 1].start() if i + 1 < len(bands) else len(text)
        values = {fact for fact, _, _ in _measures(text[band.end():right]) if fact.role in roles}
        if not values and len(bands) == 1:
            values = {fact for fact, _, _ in _measures(text[:band.start()]) if fact.role in roles}
        if not values and len(outputs) == 1:
            values = outputs  # one result may explicitly be shared by adjacent bands
        if not values:
            if outputs:
                raise ValueError("Раздели диапазоны и соответствующие им величины на утверждения.")
            continue
        allowed = rows.get(_band_key(band), set())
        for fact in values:
            if {item for item in allowed if item.role == fact.role} != {fact}:
                raise ValueError("Диапазон и величина должны совпадать с одной строкой источника.")
        checked.append(band.span())
    return checked


def _limit(prefix: str, role: str) -> str | None:
    if direct := _DIRECT_LIMIT.search(prefix):
        return "не менее" if _key(direct[1]) in {"не менее", "не меньше"} else "не более"
    # A named limit controls the next measurement in its clause. Do not borrow
    # a bound across sentences, alternatives, other measured values or headings.
    clause = re.split(r"[.;:!?]|\b(?:или|однако|тогда как)\b", prefix, flags=re.I)[-1]
    named = list(_LIMIT_WORD.finditer(clause))
    if len(named) == 1 and len(clause) - named[0].end() < 150:
        noun = named[0]["noun"]
        duration = noun in {"продолжительност", "длительност", "срок"} or noun.endswith("период")
        if duration and role not in {
            "минута", "час", "день", "месяц", "год",
        }:
            return None
        if noun in {"сумм", "стоимост", "цен"} and role != "рубль":
            return None
        return "не более" if named[0].group("upper") else "не менее"
    return None


_TEMPORAL_ANCHOR = re.compile(
    r"\b(?:(?P<expiry>по\s+(?:истечении|прошествии)|после\s+истечения|спустя|через)"
    r"|(?P<window>в\s+течение|на\s+протяжении))\s*$", re.I,
)


def _temporal_anchor(prefix: str, role: str) -> str | None:
    if role in {"минута", "час", "день", "месяц", "год"}:
        if match := _TEMPORAL_ANCHOR.search(prefix):
            return match.lastgroup
    return None


def _price_terms(quote: str, price: NumericFact) -> set[str | None]:
    """Look up the rightmost price under its nearest currency-bearing table heading."""
    lines = quote.splitlines()
    terms: set[str | None] = set()
    for i, line in enumerate(lines):
        if not _PRICE_COLUMN.search(line):
            continue
        # Table titles may wrap to a second line. Stop at the nearest table or
        # price title so a preceding table cannot lend its terms to the next one.
        above = lines[max(0, i-3):i]
        start = next((j for j in reversed(range(len(above))) if re.search(
            r"таблиц[аы]\s*№?\s*\d|(?:стоимость|цена)\s+\w+", above[j], re.I
        )), None)
        header = " ".join(above[start:]).lower() if start is not None else ""
        kinds = [k for k in _PAYMENT if k in header]
        for item in lines[i + 1:]:
            normalized = _AMOUNT.sub(lambda m: "".join(m[0].split()), item)
            amount = _ROW_PRICE.search(normalized)
            if not amount:
                break
            if _amount_value(amount[1]) == price.value:
                terms.update(kinds or [None])
    return terms


def qualify_numeric_text(text: str, quotes: list[str]) -> str:
    """Add a proven qualifier to a selected value before semantic verification.

    Conflicting bounds or payment terms need a new draft with separate claims.
    The verifier sees the revised wording and the very same citations.
    """
    matched_bands = _check_range_rows(text, quotes)
    evidence: dict[NumericFact, set[tuple[str | None, str | None]]] = {}
    for quote in quotes:
        previous = 0
        for fact, start, end in _measures(quote):
            before = quote[previous:start]
            evidence.setdefault(fact, set()).add(
                (_temporal_anchor(before, fact.role), _limit(before, fact.role))
            )
            previous = end
    previous = 0
    inserts: list[tuple[int, str]] = []
    payments: set[str | None] = set()
    amounts: set[NumericFact] = set()
    for fact, start, end in _measures(text):
        own = _limit(text[previous:start], fact.role)
        # A range endpoint was verified together with its own row value. Another
        # row may use this number as the opposite bound ("до 1"/"свыше 1").
        uses = (set() if any(a <= start < b for a, b in matched_bands)
                else evidence.get(fact, set()))
        anchor = _temporal_anchor(text[previous:start], fact.role)
        limits = {limit for _, limit in uses}
        if own and (None, own) in uses and all(ref or limit == own for ref, limit in uses):
            # An independently cited time window does not contradict the
            # explicitly stated bound on the same numerical duration.
            limits = {limit for ref, limit in uses if ref is None}
        if anchor and uses and (any(limits) or all(ref is not None for ref, _ in uses)):
            # An expiry time and a maximum duration may use the same number. A
            # bound on possession does not by itself establish a date of transfer.
            limits = {limit for ref, limit in uses if ref == anchor}
            if not limits:
                raise ValueError("Цитата не подтверждает отсчёт этого события от названного срока. "
                                 "Передай отдельно исходный срок с его пределом и условие события.")
        if any(limits):
            if len(limits) != 1:
                raise ValueError("У величины разные условия. Раздели утверждения и доказательства.")
            limit = next(iter(limits))
            if (own and own != limit) or re.search(r"\bровно\s*$", text[previous:start], re.I):
                raise ValueError(f"Сохрани исходный предел {limit} {fact.value} ({fact.role}).")
            if limit and not own:
                inserts.append((start, limit + " "))
        if fact.role == "рубль":
            amounts.add(fact)
            payments.update(term for q in quotes for term in _price_terms(q, fact))
        previous = end
    for start, words in reversed(inserts):
        text = text[:start] + words + text[start:]
    if any(payments):
        stated = {term for term in _PAYMENT if term in text.lower()}
        if len(payments) > 1 or len(amounts) > 1 or (stated and stated != payments):
            raise ValueError("Свяжи каждую цену и форму оплаты с её строкой таблицы.")
        if not stated:
            if len(list(_measures(text))) != 1:
                raise ValueError("Выдели цену и способ оплаты в отдельное утверждение.")
            term = next(k for k in payments if k)
            text = text.rstrip(" .") + " " + _PAYMENT[term] + "."
    return text
