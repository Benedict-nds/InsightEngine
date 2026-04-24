#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# faiss_index.py — build, persist, and query a FAISS ``IndexFlatIP`` vector store.
#
# Design (presentation-friendly):
#     - Vectors are **L2-normalized** (see embedder). For unit vectors, **inner
#       product equals cosine similarity** in ``[-1, 1]``.
#     - We use ``faiss.IndexFlatIP`` (inner product) — no training step, exact search.
#     - Row ``i`` in the index corresponds to ``manifest["chunks"][i]`` — a frozen
#       snapshot of ``chunk_id``, ``text``, ``metadata``, etc.
#
# Assumptions about chunks (aligned with ``ChunkRecord`` in ``chunker.py``):
#     Each chunk provides at minimum:
#         - ``chunk_id: str``
#         - ``parent_doc_id: str``
#         - ``text: str``
#         - ``metadata: dict`` (JSON-serializable values recommended)
#
# Optional fields preserved when present on dataclass or dict:
#         - ``source_kind``, ``chunk_index``, ``char_start``, ``char_end``
#


from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np

from src.utils.helpers import ensure_dir, load_json, save_json


INDEX_FILENAME = "faiss_index_flatip.faiss"
MANIFEST_FILENAME = "faiss_chunk_manifest.json"


def chunk_record_to_manifest_dict(chunk: Any) -> dict[str, Any]:
    # Convert a :class:`~src.chunking.chunker.ChunkRecord` or dict to a JSON-safe dict.
    # The manifest is the single source of truth mapping FAISS row → chunk fields
    # for debugging and UI display.

    if is_dataclass(chunk):
        return asdict(chunk)
    if isinstance(chunk, Mapping):
        return dict(chunk)
    raise TypeError("chunk must be a dataclass instance or mapping")


class FaissChunkIndex:
    # FAISS inner-product index over L2-normalized chunk embeddings.
    # Use :meth:`build` then :meth:`save`, or :meth:`load` to restore from disk.


    def __init__(self, index: Any, chunks_manifest: list[dict[str, Any]]) -> None:
        self._index = index
        self._chunks: list[dict[str, Any]] = list(chunks_manifest)
        self._validate_alignment()

    @property
    def embedding_dim(self) -> int:
        return int(self._index.d)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def _validate_alignment(self) -> None:
        n_index = int(self._index.ntotal)
        n_manifest = len(self._chunks)
        if n_index != n_manifest:
            raise ValueError(
                f"FAISS rows ({n_index}) and manifest chunks ({n_manifest}) must match"
            )

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunks: Sequence[Any],
    ) -> FaissChunkIndex:
        # Create a new flat inner-product index from embeddings and parallel chunks.
        # Args:
        # embeddings: ``float32`` array, shape ``(n, d)``. Rows should be
        # L2-normalized for cosine-style scores via inner product.
        # chunks: Length-``n`` sequence in **the same row order** as embeddings.
        # Returns:
        # Ready-to-search :class:`FaissChunkIndex`.

        matrix = np.asarray(embeddings, dtype=np.float32, order="C")
        if matrix.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {matrix.shape}")
        n, d = matrix.shape
        if len(chunks) != n:
            raise ValueError(
                f"chunks length ({len(chunks)}) must match embeddings rows ({n})"
            )

        # Ensure L2-normalized rows (safe if embedder already normalized)
        faiss.normalize_L2(matrix)

        manifest = [chunk_record_to_manifest_dict(c) for c in chunks]
        index = faiss.IndexFlatIP(d)
        index.add(matrix)
        return cls(index, manifest)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        # Retrieve top-k chunks by inner product (cosine similarity if normalized).
        # Args:
        # query_vector: Shape ``(1, d)`` or ``(d,)`` — float32 preferred.
        # top_k: Number of neighbors (clamped to index size).
        # Returns:
        # List of hit dicts, sorted best-first, each containing:
        # ``faiss_row``, ``score`` (inner product / cosine in [-1,1] for unit vectors),
        # ``chunk_id``, ``parent_doc_id``, ``text``, ``metadata``, plus any
        # extra fields stored in the manifest.

        if self._index.ntotal == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"query_vector must be (1, d) or (d,), got {q.shape}")
        if q.shape[1] != self.embedding_dim:
            raise ValueError(
                f"query dim {q.shape[1]} != index dim {self.embedding_dim}"
            )

        faiss.normalize_L2(q)
        k = min(int(top_k), int(self._index.ntotal))
        scores, indices = self._index.search(q, k)

        hits: list[dict[str, Any]] = []
        for rank, (score, row_idx) in enumerate(zip(scores[0].tolist(), indices[0].tolist())):
            if row_idx < 0:
                continue
            chunk = dict(self._chunks[int(row_idx)])
            hit = {
                "rank": rank,
                "faiss_row": int(row_idx),
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "parent_doc_id": chunk.get("parent_doc_id"),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
            }
            # Preserve optional presentation fields if present in manifest
            for key in ("source_kind", "chunk_index", "char_start", "char_end"):
                if key in chunk:
                    hit[key] = chunk[key]
            hits.append(hit)
        return hits

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        # Persist FAISS index binary + JSON manifest (chunk rows in FAISS order).
        # Returns:
        # Tuple ``(index_path, manifest_path)``.

        d = ensure_dir(directory)
        ipath = d / INDEX_FILENAME
        mpath = d / MANIFEST_FILENAME
        faiss.write_index(self._index, str(ipath))
        save_json(
            {
                "embedding_dim": self.embedding_dim,
                "ntotal": self.ntotal,
                "index_type": "IndexFlatIP",
                "similarity": "inner_product_on_l2_normalized_vectors_equals_cosine",
                "chunks": self._chunks,
            },
            mpath,
        )
        return ipath.resolve(), mpath.resolve()

    @classmethod
    def load(cls, directory: str | Path) -> FaissChunkIndex:
        # Load index + manifest produced by :meth:`save`.
        # Args:
        # directory: Folder containing ``faiss_index_flatip.faiss`` and manifest JSON.

        d = Path(directory)
        ipath = d / INDEX_FILENAME
        mpath = d / MANIFEST_FILENAME
        if not ipath.is_file():
            raise FileNotFoundError(f"Missing FAISS index file: {ipath}")
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing manifest file: {mpath}")

        index = faiss.read_index(str(ipath))
        payload = load_json(mpath)
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("manifest missing 'chunks' list")
        return cls(index, chunks)
