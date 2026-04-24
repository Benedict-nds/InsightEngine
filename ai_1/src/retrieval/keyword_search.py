#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# keyword_search.py — transparent lexical scoring for hybrid retrieval.
#
# Approach (intentionally simple for viva / slides):
#     1. **Tokenize** query and document into lowercase alphanumeric tokens
#        (no stemming — predictable behavior).
#     2. **Overlap score**: Jaccard-like ``|Q ∩ D| / |Q ∪ D|`` capped at 1.
#     3. **Coverage score**: fraction of distinct query tokens found in the document.
#     4. **Exact phrase rescue**: small bonus if the full normalized query substring
#        appears in the original chunk text (helps names, years, codes).
#     5. **Structured bonus** (election CSV rows): year / region / high-vote alignment
#        so queries like "who won in 2020" prefer rows with large vote counts.
#     6. **Numeric signal bonus** for budget / quantity questions (PDF + tables).
#     7. **Transport keyword bonus** when ``road`` / ``transport`` / … appear in
#        both query and chunk (small, capped).
#     8. **Money signal bonus** when the query implies budget/money and the chunk
#        shows currency-scale amounts (GH¢, $, ``N million``, …) rather than bare counts.
#
# The **base** lexical score stays in ``[0, 1]``. The public :func:`keyword_score`
# returns base + structured + numeric + transport + money bonuses (may exceed 1.0).
#


from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Pipe-separated row text uses ``votes: 12345`` style (from load_csv row_to_text).
_VOTES_RE = re.compile(r"votes:\s*([0-9]+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"year:\s*(\d{4})", re.IGNORECASE)
_REGION_RE = re.compile(r"region:\s*([^|]+)", re.IGNORECASE)

# Substrings for region match between query and chunk (Ghana regions; keep explicit).
_GHANA_REGION_PHRASES: tuple[str, ...] = (
    "greater accra region",
    "greater accra",
    "ashanti region",
    "ashanti",
    "western north region",
    "western north",
    "western region",
    "western",
    "eastern region",
    "eastern",
    "central region",
    "central",
    "volta region",
    "volta",
    "northern region",
    "northern",
    "upper east region",
    "upper east",
    "upper west region",
    "upper west",
    "bono region",
    "bono",
    "bono east region",
    "bono east",
    "ahafo region",
    "ahafo",
    "oti region",
    "oti",
    "savannah region",
    "savannah",
    "north east region",
    "north east",
)

# Budget / quantity phrasing → prefer chunks that actually contain numbers.
_NUMERIC_INTENT_KEYWORDS: tuple[str, ...] = (
    "how much",
    "amount",
    "budget",
    "allocation",
    "cost",
    "total",
    "number",
    "value",
)

# Digit-led tokens: ``123``, ``1,234.56``, ``2025`` (multiple matches OK).
_NUMBER_TOKEN_RE = re.compile(r"\d[\d,\.]*")

# Small extra signal when user and chunk both mention infrastructure terms.
_TRANSPORT_TERMS: tuple[str, ...] = (
    "road",
    "roads",
    "transport",
    "highways",
    "highway",
)

# Budget / money phrasing (subset of numeric intent — stricter for currency boost).
_MONEY_INTENT_KEYWORDS: tuple[str, ...] = (
    "budget",
    "allocation",
    "allocate",
    "allocated",
    "cost",
    "amount",
    "fund",
    "spending",
    "how much",
)

_MONEY_CURRENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"gh[c¢]\s?\d[\d,\.]*", re.IGNORECASE),
    re.compile(r"\$\s?\d[\d,\.]*"),
    re.compile(r"\d[\d,\.]*\s?(billion|million|thousand)"),
)


_WINNER_INTENT_PHRASES: tuple[str, ...] = (
    "won",
    "winner",
    "highest",
    "most votes",
    "who won",
    "win the",
)


def extract_votes(text: str) -> int:
    # Parse vote count from flattened CSV row text.
    # Expects a fragment like ``votes: 658626`` (see ``row_to_text`` in load_csv).
    # Returns:
    # Integer vote count, or ``0`` if not found / not parseable.

    if not text:
        return 0
    m = _VOTES_RE.search(text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def extract_year(text: str) -> str:
    # Return election year from chunk text, e.g. ``"2020"``.
    # Looks for ``year: 2020`` first; otherwise returns empty string.

    if not text:
        return ""
    m = _YEAR_RE.search(text)
    return m.group(1) if m else ""


def extract_region(text: str) -> str:
    # Return region field from chunk text, e.g. ``"Greater Accra Region"``.
    # Uses ``region: ... |`` from row formatting; stripped, may be empty.

    if not text:
        return ""
    m = _REGION_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def structured_bonus(query: str, chunk_text: str) -> float:
    # Additive, rule-based signal for Ghana election row chunks (explainable in slides).
    # Rules (cumulative):
    # 1. **Year match:** query mentions a 4-digit year that appears in the chunk → +0.5.
    # 2. **Region match:** query mentions a known Ghana region phrase that also
    # appears in the chunk (case-insensitive) → +0.5.
    # 3. **Winner / magnitude:** if query shows "winner" intent, boost rows with
    # large ``votes`` (parsed): >1_000_000 → +1.0; >100_000 → +0.5.
    # 4. **Year mismatch penalty:** query mentions at least one 4-digit year but
    # **none** of those year strings appear in the chunk → -0.3.
    # Returns:
    # Total bonus (may be negative).

    q = (query or "").lower()
    c = (chunk_text or "").lower()
    bonus = 0.0

    # --- Year tokens in query (simple 19xx / 20xx) ---
    years_in_query = set(re.findall(r"\b(19\d{2}|20\d{2})\b", query or ""))
    matched_any_year = any(yr in c for yr in years_in_query)
    if matched_any_year:
        bonus += 0.5
    elif years_in_query:
        # Query mentions a year but this row shows no matching year string in text
        bonus -= 0.3

    # --- Region: longest phrases first to prefer specific matches ---
    for phrase in sorted(_GHANA_REGION_PHRASES, key=len, reverse=True):
        if phrase in q and phrase in c:
            bonus += 0.5
            break

    # --- Winner intent + vote magnitude ---
    winner_intent = any(w in q for w in _WINNER_INTENT_PHRASES)
    if winner_intent:
        votes = extract_votes(chunk_text)
        if votes > 1_000_000:
            bonus += 1.0
        elif votes > 100_000:
            bonus += 0.5

    return float(bonus)


def numeric_token_count(text: str) -> int:
        # Count digit-led numeric tokens in ``text`` (same rule as :func:`numeric_signal_bonus`).

    return len(_NUMBER_TOKEN_RE.findall(text or ""))


def numeric_signal_bonus(query: str, text: str) -> float:
    # Boost chunks that contain numeric tokens when the query implies quantities.
    # If numeric intent is detected but the chunk has no digit-like tokens, apply
    # a small penalty so purely narrative PDF slices sink below tables with figures.

    q = (query or "").lower()
    has_numeric_intent = any(k in q for k in _NUMERIC_INTENT_KEYWORDS)
    if not has_numeric_intent:
        return 0.0

    numbers = _NUMBER_TOKEN_RE.findall(text or "")
    if not numbers:
        return -0.2
    return min(0.5, 0.1 * len(numbers))


def transport_keyword_bonus(query: str, text: str) -> float:
    # Small additive bonus when infrastructure terms appear in **both** query and chunk.
    # Capped so this stays a tie-breaker, not a second retrieval model.

    q = (query or "").lower()
    c = (text or "").lower()
    bonus = 0.0
    for term in _TRANSPORT_TERMS:
        if term in q and term in c:
            bonus += 0.1
    return float(min(0.3, bonus))


def money_matches_in_text(text: str) -> list[str]:
    # Collect substrings that look like **financial** amounts (currency or scale words).
    # Used for :func:`money_signal_bonus` and for ``money_matches`` debug fields.
    # Order is pattern order then match order; duplicates are removed while keeping
    # first occurrence.

    t = (text or "").lower()
    raw: list[str] = []
    for pat in _MONEY_CURRENCY_PATTERNS:
        raw.extend(pat.findall(t))
    seen: set[str] = set()
    out: list[str] = []
    for m in raw:
        s = str(m).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def money_signal_bonus(query: str, text: str) -> float:
    # Boost chunks that contain financial-looking values when the query implies money.
    # If money intent is present but no currency / scale pattern matches, returns a
    # small **penalty** so chunks with only ``km``, ``%``, or bare counts stay below
    # rows that actually quote allocations.

    q = (query or "").lower()
    has_money_intent = any(k in q for k in _MONEY_INTENT_KEYWORDS)
    if not has_money_intent:
        return 0.0

    matches = money_matches_in_text(text)
    if not matches:
        return -0.3
    return float(min(1.0, 0.4 * len(matches)))


def tokenize(text: str) -> list[str]:
    # Lowercase alphanumeric tokens only (no stopword list — explicit tradeoff).
    # Args:
    # text: Raw chunk or query string.
    # Returns:
    # Token list (may contain duplicates; overlap uses sets).

    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def jaccard_like_overlap(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    # ``|Q ∩ D| / |Q ∪ D|`` with safe handling of empty sets.

    q_set = set(query_tokens)
    d_set = set(doc_tokens)
    if not q_set and not d_set:
        return 0.0
    inter = q_set & d_set
    union = q_set | d_set
    if not union:
        return 0.0
    return float(len(inter)) / float(len(union))


def query_coverage(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    # Fraction of **distinct** query types that appear in the document at least once.
    # If the query has no tokens, returns ``0.0``.

    q_set = set(query_tokens)
    if not q_set:
        return 0.0
    d_set = set(doc_tokens)
    hits = len(q_set & d_set)
    return float(hits) / float(len(q_set))


def exact_query_substring_bonus(query: str, chunk_text: str) -> float:
    # Return ``1.0`` if the trimmed lowercase query appears as a substring in the
    # lowered chunk text, else ``0.0``. Short queries (< 2 chars after strip)
    # return ``0.0`` to avoid noise.

    q = (query or "").strip().lower()
    if len(q) < 2:
        return 0.0
    hay = (chunk_text or "").lower()
    return 1.0 if q in hay else 0.0


def keyword_base_score(
    query: str,
    chunk_text: str,
    *,
    w_overlap: float = 0.45,
    w_coverage: float = 0.35,
    w_exact: float = 0.20,
) -> float:
    # Original lexical blend only (overlap + coverage + exact), clamped to ``[0, 1]``.
    # This is the **base** part before structured election bonuses.

    qt = tokenize(query)
    dt = tokenize(chunk_text)
    overlap = jaccard_like_overlap(qt, dt)
    cover = query_coverage(qt, dt)
    exact = exact_query_substring_bonus(query, chunk_text)

    w_sum = w_overlap + w_coverage + w_exact
    if w_sum <= 0:
        return 0.0
    raw = (w_overlap * overlap + w_coverage * cover + w_exact * exact) / w_sum
    return float(max(0.0, min(1.0, raw)))


def keyword_score(
    query: str,
    chunk_text: str,
    *,
    w_overlap: float = 0.45,
    w_coverage: float = 0.35,
    w_exact: float = 0.20,
) -> float:
    # Lexical base plus structured, numeric, transport, and money bonuses (not clamped to 1).
    # Args:
    # query: User query string.
    # chunk_text: Chunk body to score against.
    # w_overlap: Weight for Jaccard-like token overlap.
    # w_coverage: Weight for query-token coverage in the chunk.
    # w_exact: Weight for exact normalized substring match.
    # Returns:
    # ``keyword_base_score + structured_bonus + numeric_signal_bonus
    # + transport_keyword_bonus + money_signal_bonus`` (may exceed 1.0 for ranking).

    base = keyword_base_score(
        query, chunk_text, w_overlap=w_overlap, w_coverage=w_coverage, w_exact=w_exact
    )
    sb = structured_bonus(query, chunk_text)
    nb = numeric_signal_bonus(query, chunk_text)
    tb = transport_keyword_bonus(query, chunk_text)
    mb = money_signal_bonus(query, chunk_text)
    return float(base + sb + nb + tb + mb)


@dataclass
class KeywordHit:
        # Structured keyword score for one chunk (debuggable).


    chunk_id: str
    score: float
    overlap: float
    coverage: float
    exact_substring: float
    tokens_query: int
    tokens_chunk: int
    keyword_base_score: float
    structured_bonus: float
    numeric_bonus: float
    numbers_found: int
    transport_keyword_bonus: float
    money_bonus: float
    money_matches: list[str]
    votes_extracted: int
    year_extracted: str
    region_extracted: str


def score_chunk_debug(query: str, chunk_id: str, chunk_text: str) -> KeywordHit:
    # Same scoring as :func:`keyword_score`, but return intermediate numbers for UI/logs.

    qt = tokenize(query)
    dt = tokenize(chunk_text)
    ov = jaccard_like_overlap(qt, dt)
    cov = query_coverage(qt, dt)
    ex = exact_query_substring_bonus(query, chunk_text)
    base = keyword_base_score(query, chunk_text)
    sb = structured_bonus(query, chunk_text)
    nb = numeric_signal_bonus(query, chunk_text)
    tb = transport_keyword_bonus(query, chunk_text)
    mm = money_matches_in_text(chunk_text)
    mb = money_signal_bonus(query, chunk_text)
    nums = numeric_token_count(chunk_text)
    full = base + sb + nb + tb + mb
    return KeywordHit(
        chunk_id=chunk_id,
        score=full,
        overlap=ov,
        coverage=cov,
        exact_substring=ex,
        tokens_query=len(set(qt)),
        tokens_chunk=len(set(dt)),
        keyword_base_score=base,
        structured_bonus=sb,
        numeric_bonus=nb,
        numbers_found=nums,
        transport_keyword_bonus=tb,
        money_bonus=mb,
        money_matches=list(mm),
        votes_extracted=extract_votes(chunk_text),
        year_extracted=extract_year(chunk_text),
        region_extracted=extract_region(chunk_text),
    )


def rank_chunks_by_keyword(
    query: str,
    chunks: Sequence[Any],
    *,
    text_key: str = "text",
    id_key: str = "chunk_id",
    top_k: int = 20,
) -> list[dict[str, Any]]:
    # Score every chunk in ``chunks`` and return the top ``top_k`` by keyword score.
    # ``chunks`` may be manifest dicts or :class:`~src.chunking.chunker.ChunkRecord`
    # objects.
    # Returns:
    # List of dicts sorted best-first, including exam-oriented debug fields:
    # ``votes_extracted``, ``year_extracted``, ``region_extracted``,
    # ``structured_bonus``, ``keyword_base_score``, ``numeric_bonus``,
    # ``numbers_found``, ``transport_keyword_bonus``, ``money_bonus``,
    # ``money_matches``.

    scored: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        if isinstance(c, dict):
            cid = str(c.get(id_key, ""))
            text = str(c.get(text_key, ""))
        else:
            cid = str(getattr(c, id_key))
            text = str(getattr(c, text_key))
        hit = score_chunk_debug(query, cid, text)
        scored.append(
            (
                hit.score,
                {
                    "chunk_id": hit.chunk_id,
                    "keyword_score": hit.score,
                    "keyword_base_score": hit.keyword_base_score,
                    "structured_bonus": hit.structured_bonus,
                    "numeric_bonus": hit.numeric_bonus,
                    "numbers_found": hit.numbers_found,
                    "number_count": hit.numbers_found,
                    "transport_keyword_bonus": hit.transport_keyword_bonus,
                    "money_bonus": hit.money_bonus,
                    "money_matches": list(hit.money_matches),
                    "votes_extracted": hit.votes_extracted,
                    "year_extracted": hit.year_extracted,
                    "region_extracted": hit.region_extracted,
                    "keyword_overlap": hit.overlap,
                    "keyword_coverage": hit.coverage,
                    "keyword_exact_substring": hit.exact_substring,
                    "keyword_tokens_query": hit.tokens_query,
                    "keyword_tokens_chunk": hit.tokens_chunk,
                    "text": text,
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for rank, (sc, payload) in enumerate(scored[: max(0, int(top_k))]):
        row = dict(payload)
        row["keyword_rank"] = rank
        out.append(row)
    return out
