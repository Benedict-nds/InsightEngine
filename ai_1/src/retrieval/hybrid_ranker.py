#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# hybrid_ranker.py — fuse dense vector similarity with lexical + structured scores.
#
# Why hybrid helps (exam talking points):
#     - **Dense retrieval** excels at paraphrases and conceptual overlap but can
#       miss rare **exact tokens** (candidate names, constituency spellings,
#       budget line codes) when the embedding space underweights them.
#     - **Keyword scoring** is brittle alone, but as a *second signal* it rescues
#       chunks that literally contain the query tokens even when cosine similarity
#       is middling.
#     - **Structured bonus** (year / region / high-vote rows) improves tabular
#       election queries without any extra ML libraries — rules you can justify
#       on a slide.
#     - **Numeric + transport + money bonuses** (from :mod:`src.retrieval.keyword_search`)
#       lift chunks that contain figures, currency-style amounts, and matching domain terms.
#
# Final fusion (additive signals not scaled by ``beta``)::
#
#     final_score =
#         alpha * vector_score_01
#         + beta * keyword_base_score
#         + structured_bonus(query, chunk_text)
#         + numeric_signal_bonus(query, chunk_text)
#         + transport_keyword_bonus(query, chunk_text)
#         + money_signal_bonus(query, chunk_text)
#
# Here ``keyword_base_score`` is the original ``[0, 1]`` lexical blend from
# :mod:`src.retrieval.keyword_search`. Structured, numeric, transport, and money
# terms are imported once (no duplicated logic) and are **not** multiplied by
# ``beta`` so salient facts stay visible when ``beta`` is small.
#
# Implementation notes:
#     - Vector scores are inner products in ``[-1, 1]`` for L2-normalized vectors;
#       we map them to ``[0, 1]`` via ``(cosine + 1) / 2`` for blending with the
#       base keyword score in ``[0, 1]``.
#     - We build a **candidate pool** from vector top-``vector_pool_k`` hits, then
#       optionally union keyword top-``keyword_pool_k`` from the full corpus
#       (requires passing ``all_chunks`` for the lexical pass).
#     - Each output row includes ``votes_extracted``, ``year_extracted``,
#       ``region_extracted``, ``structured_bonus``, ``numeric_bonus``,
#       ``numbers_found``, ``transport_keyword_bonus``, ``money_bonus``,
#       ``money_matches``, and ``keyword_base_score``.
#


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.retrieval.keyword_search import (
    extract_region,
    extract_votes,
    extract_year,
    keyword_base_score,
    keyword_score,
    money_matches_in_text,
    money_signal_bonus,
    numeric_signal_bonus,
    numeric_token_count,
    rank_chunks_by_keyword,
    structured_bonus,
    transport_keyword_bonus,
)
from src.retrieval.retriever import VectorRetriever, cosine_to_unit_interval


@dataclass
class HybridConfig:
        # Weights and pool sizes for hybrid fusion.


    alpha: float = 0.7
    beta: float = 0.3
    vector_pool_k: int = 20
    keyword_pool_k: int = 20
    final_top_k: int = 5


def _chunk_text_and_id(chunk: Any) -> tuple[str, str]:
    if isinstance(chunk, Mapping):
        return str(chunk.get("text", "")), str(chunk.get("chunk_id", ""))
    return str(getattr(chunk, "text", "")), str(getattr(chunk, "chunk_id", ""))


def _index_chunks_by_id(chunks: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in chunks:
        if isinstance(c, Mapping):
            cid = str(c.get("chunk_id", ""))
        else:
            cid = str(getattr(c, "chunk_id", ""))
        if cid:
            out[cid] = c
    return out


def merge_hybrid_scores(
    query: str,
    vector_hits: Sequence[Mapping[str, Any]],
    *,
    all_chunks: Sequence[Any] | None = None,
    config: HybridConfig | None = None,
) -> list[dict[str, Any]]:
    # Combine precomputed vector hits with lexical base, structured bonus, and vector term.
    # Args:
    # query: User query string.
    # vector_hits: Output from :meth:`VectorRetriever.retrieve` (each has ``chunk_id``,
    # ``score`` cosine/IP, ``text``, ``metadata``, etc.).
    # all_chunks: Full chunk list (manifest dicts or ``ChunkRecord``) for keyword
    # ranking over the corpus. If ``None``, keyword signal is only computed
    # for chunk texts present in ``vector_hits`` (still useful, narrower pool).
    # config: Blending weights / pool sizes.
    # Returns:
    # Sorted list of enriched hit dicts with ``final_score``, debug extractions,
    # and ``structured_bonus`` echoed for logging / Streamlit.

    cfg = config or HybridConfig()
    if cfg.alpha < 0 or cfg.beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    # Candidate chunk_ids: top vector pool + optional keyword rescues from full corpus
    vec_pool = list(vector_hits)[: max(0, int(cfg.vector_pool_k))]
    cand_ids: set[str] = set()
    for h in vec_pool:
        cid = str(h.get("chunk_id", ""))
        if cid:
            cand_ids.add(cid)

    kw_rows: list[dict[str, Any]] = []
    if all_chunks is not None:
        kw_rows = rank_chunks_by_keyword(
            query,
            list(all_chunks),
            top_k=int(cfg.keyword_pool_k),
        )
        for row in kw_rows:
            cid = str(row.get("chunk_id", ""))
            if cid:
                cand_ids.add(cid)

    by_id = _index_chunks_by_id(all_chunks) if all_chunks is not None else {}

    # Precompute full keyword_score for keyword-only ranking / fallback sort keys
    kw_score_by_id: dict[str, float] = {
        row["chunk_id"]: float(row["keyword_score"]) for row in kw_rows
    }

    merged: list[dict[str, Any]] = []
    for cid in cand_ids:
        base: dict[str, Any]
        vec_hit = next((dict(h) for h in vec_pool if str(h.get("chunk_id")) == cid), None)
        if cid in by_id:
            c = by_id[cid]
            text, _ = _chunk_text_and_id(c)
            if isinstance(c, Mapping):
                base = dict(c)
            else:
                from dataclasses import asdict, is_dataclass

                base = asdict(c) if is_dataclass(c) else dict(vec_hit or {})
        elif vec_hit is not None:
            base = dict(vec_hit)
            text = str(base.get("text", ""))
        else:
            kw_row = next((dict(r) for r in kw_rows if str(r.get("chunk_id")) == cid), None)
            if kw_row is not None:
                text = str(kw_row.get("text", ""))
                base = {
                    "chunk_id": cid,
                    "text": text,
                    "metadata": {},
                }
            else:
                text = ""
                base = {"chunk_id": cid, "text": "", "metadata": {}}

        if cid not in kw_score_by_id:
            kw_score_by_id[cid] = keyword_score(query, text)

        v_raw = float(vec_hit["score"]) if vec_hit and "score" in vec_hit else -1.0
        v_01 = cosine_to_unit_interval(v_raw) if vec_hit and "score" in vec_hit else 0.0

        k_base = keyword_base_score(query, text)
        sb = structured_bonus(query, text)
        nb = numeric_signal_bonus(query, text)
        tb = transport_keyword_bonus(query, text)
        mb = money_signal_bonus(query, text)
        mm = money_matches_in_text(text)
        final = float(cfg.alpha) * v_01 + float(cfg.beta) * k_base + sb + nb + tb + mb

        votes_ex = extract_votes(text)
        year_ex = extract_year(text)
        region_ex = extract_region(text)
        numbers_found = numeric_token_count(text)

        row = dict(base)
        row.update(
            {
                "retrieval_mode": "hybrid",
                "vector_score_raw": v_raw,
                "vector_score_01": v_01,
                "keyword_base_score": k_base,
                "keyword_score": float(kw_score_by_id[cid]),
                "structured_bonus": sb,
                "numeric_bonus": nb,
                "numbers_found": numbers_found,
                "number_count": numbers_found,
                "transport_keyword_bonus": tb,
                "money_bonus": mb,
                "money_matches": list(mm),
                "votes_extracted": votes_ex,
                "year_extracted": year_ex,
                "region_extracted": region_ex,
                "final_score": final,
                "hybrid_alpha": float(cfg.alpha),
                "hybrid_beta": float(cfg.beta),
            }
        )
        merged.append(row)

    merged.sort(
        key=lambda r: (float(r.get("final_score", 0.0)), float(r.get("vector_score_raw", -1.0))),
        reverse=True,
    )
    return merged[: max(0, int(cfg.final_top_k))]


class HybridRetriever:
    # End-to-end hybrid retrieval: vector search then keyword fusion / rerank.


    def __init__(
        self,
        vector_retriever: VectorRetriever,
        *,
        all_chunks: Sequence[Any] | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self._vr = vector_retriever
        self._all_chunks = list(all_chunks) if all_chunks is not None else None
        self._config = config or HybridConfig()

    def set_corpus_chunks(self, chunks: Sequence[Any] | None) -> None:
                # Attach the full chunk list for global keyword rescoring (optional).

        self._all_chunks = list(chunks) if chunks is not None else None

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        # Retrieve using dense vectors, then rerank with hybrid scores.
        # ``top_k`` here maps to :class:`HybridConfig.final_top_k` when provided,
        # leaving pool sizes from ``HybridConfig`` unless you construct a fresh config.

        cfg = self._config
        if top_k is not None:
            cfg = HybridConfig(
                alpha=cfg.alpha,
                beta=cfg.beta,
                vector_pool_k=max(int(top_k), cfg.vector_pool_k),
                keyword_pool_k=cfg.keyword_pool_k,
                final_top_k=int(top_k),
            )

        vec_hits = self._vr.retrieve(
            query,
            top_k=max(cfg.vector_pool_k, cfg.final_top_k),
            include_score_unit_interval=True,
        )
        return merge_hybrid_scores(
            query,
            vec_hits,
            all_chunks=self._all_chunks,
            config=cfg,
        )
