#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# retriever.py — baseline **vector-only** retrieval using embedder + FAISS.
#
# Pipeline (explicit for demos):
#     1. Embed the query with :class:`~src.embedding.embedder.SentenceTransformerEmbedder`.
#     2. Search the :class:`~src.retrieval.faiss_index.FaissChunkIndex` for ``top_k``.
#     3. Return structured dicts suitable for logging, Streamlit, or hybrid reranking.
#
# Scores are **inner products** of L2-normalized vectors (= cosine similarity in
# ``[-1, 1]``). For UI-friendly ``[0, 1]`` scaling, see ``cosine_to_unit_interval``.
#


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.embedding.embedder import SentenceTransformerEmbedder
from src.retrieval.faiss_index import FaissChunkIndex


def cosine_to_unit_interval(cosine_sim: float) -> float:
    # Map cosine / inner-product score from ``[-1, 1]`` to ``[0, 1]`` for blending.
    # This is a linear display transform only — it does not change ranking order
    # among a fixed set of cosine scores.

    return float(max(0.0, min(1.0, (float(cosine_sim) + 1.0) / 2.0)))


@dataclass
class VectorRetrievalConfig:
        # Tunable knobs for vector retrieval (kept small for clarity).


    top_k: int = 5


class VectorRetriever:
    # Thin orchestration layer: embedder + FAISS index → ranked chunk hits.


    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        index: FaissChunkIndex,
        *,
        config: VectorRetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._config = config or VectorRetrievalConfig()

        if self._embedder.embedding_dim != self._index.embedding_dim:
            raise ValueError(
                f"Embedder dim {self._embedder.embedding_dim} != index dim {self._index.embedding_dim}"
            )

    @property
    def embedder(self) -> SentenceTransformerEmbedder:
        return self._embedder

    @property
    def index(self) -> FaissChunkIndex:
        return self._index

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_score_unit_interval: bool = True,
    ) -> list[dict[str, Any]]:
        # Run vector-only top-k retrieval for a natural language query.
        # Args:
        # query: User question string.
        # top_k: Override for number of hits; defaults to ``VectorRetrievalConfig.top_k``.
        # include_score_unit_interval: If True, add ``score_01`` = mapped cosine to [0,1].
        # Returns:
        # List of hit dictionaries (best first). Each hit includes at minimum:
        # ``rank``, ``faiss_row``, ``score`` (cosine/IP), ``chunk_id``,
        # ``parent_doc_id``, ``text``, ``metadata``, plus optional chunk fields
        # echoed from the FAISS manifest.

        k = int(top_k) if top_k is not None else int(self._config.top_k)
        if k < 1:
            raise ValueError("top_k must be at least 1")

        qvec = self._embedder.embed_query(query)
        raw_hits = self._index.search(qvec, k)

        enriched: list[dict[str, Any]] = []
        for h in raw_hits:
            hit = dict(h)
            if include_score_unit_interval and "score" in hit:
                hit["score_01"] = cosine_to_unit_interval(float(hit["score"]))
            hit["retrieval_mode"] = "vector"
            enriched.append(hit)
        return enriched
