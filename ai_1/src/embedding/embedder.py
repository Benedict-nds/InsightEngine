#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# embedder.py — sentence-transformers wrapper for chunk and query embeddings.
#
# Responsibilities:
#     - Load a named embedding model (default: all-MiniLM-L6-v2, 384-d).
#     - Encode lists of texts in batches, return float32 NumPy matrices.
#     - Encode a single query as a 2D row vector (1, dim) for FAISS search.
#     - Optionally persist embeddings next to a row-ordered manifest (chunk ids).
#
# Cosine / FAISS note:
#     Encodings use ``normalize_embeddings=True`` so each vector has unit L2
#     norm. Inner product (IndexFlatIP) between normalized vectors equals
#     cosine similarity in [-1, 1].
#
# Swapping models: pass a different ``model_name``; dimension must match any
# existing FAISS index you load (rebuild index if you change model).
#


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.utils.helpers import ensure_dir, load_json, save_json


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _chunks_to_manifest_rows(
    chunks: Sequence[Any],
) -> list[dict[str, Any]]:
    # Build minimal JSON-serializable rows aligned with embedding row order.
    # Accepts :class:`src.chunking.chunker.ChunkRecord` or dict-like objects with
    # at least ``chunk_id`` (used to verify alignment when reloading).

    rows: list[dict[str, Any]] = []
    for c in chunks:
        if hasattr(c, "chunk_id"):
            rows.append({"chunk_id": getattr(c, "chunk_id")})
        elif isinstance(c, Mapping):
            rows.append({"chunk_id": str(c.get("chunk_id", ""))})
        else:
            raise TypeError("Each chunk must be ChunkRecord or mapping with chunk_id")
    return rows


@dataclass
class EmbeddingBatchResult:
        # Outcome of embedding a list of texts.


    vectors: np.ndarray  # shape (n, dim), float32, L2-normalized rows
    model_name: str
    batch_size: int


class SentenceTransformerEmbedder:
    # Thin, explicit wrapper around ``sentence_transformers.SentenceTransformer``.
    # Public methods return NumPy arrays (float32) suitable for FAISS.


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        default_batch_size: int = 32,
    ) -> None:
        # Args:
        # model_name: Hugging Face id or local path for the model weights.
        # device: Optional torch device string (``"cpu"``, ``"cuda"``, ...).
        # default_batch_size: Default batch size for :meth:`embed_texts`.

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._default_batch_size = int(default_batch_size)
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
                # Vector dimensionality ``d`` for FAISS ``IndexFlatIP``.

        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Embed many strings in row order (row ``i`` corresponds to ``texts[i]``).
        # Args:
        # texts: Iterable of chunk or document strings (empty strings allowed;
        # they still occupy a row — consider filtering upstream).
        # batch_size: Override for this call; defaults to constructor value.
        # show_progress_bar: Passed through to sentence-transformers (off by default).
        # Returns:
        # ``float32`` array of shape ``(len(texts), embedding_dim)``, each row
        # L2-normalized when ``normalize_embeddings`` is enabled below.

        bs = int(batch_size) if batch_size is not None else self._default_batch_size
        if bs < 1:
            raise ValueError("batch_size must be at least 1")

        vectors = self._model.encode(
            list(texts),
            batch_size=bs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        out = np.asarray(vectors, dtype=np.float32)
        if out.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {out.shape}")
        return out

    def embed_query(self, query: str) -> np.ndarray:
        # Embed a single user query as one normalized row vector.
        # Args:
        # query: Natural language query string.
        # Returns:
        # Array of shape ``(1, embedding_dim)``, dtype float32.

        q = query if isinstance(query, str) else str(query)
        vec = self.embed_texts([q], batch_size=1)
        return vec

    def embed_chunks(
        self,
        chunks: Sequence[Any],
        *,
        text_attr: str = "text",
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> EmbeddingBatchResult:
        # Convenience: pull ``text`` (or ``text_attr``) from chunk objects / dicts.
        # Args:
        # chunks: Sequence of :class:`~src.chunking.chunker.ChunkRecord` or dicts.
        # text_attr: Attribute / key name holding the chunk string.
        # batch_size: Optional batch override.
        # show_progress_bar: Progress bar toggle.
        # Returns:
        # :class:`EmbeddingBatchResult` with ``vectors`` aligned to ``chunks``.

        texts: list[str] = []
        for c in chunks:
            if hasattr(c, text_attr):
                t = getattr(c, text_attr)
            elif isinstance(c, Mapping):
                t = c.get(text_attr, "")
            else:
                raise TypeError("Chunk must be object or mapping with text field")
            texts.append("" if t is None else str(t))
        vecs = self.embed_texts(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        return EmbeddingBatchResult(
            vectors=vecs,
            model_name=self._model_name,
            batch_size=batch_size or self._default_batch_size,
        )

    def save_embeddings_bundle(
        self,
        vectors: np.ndarray,
        chunks: Sequence[Any],
        directory: str | Path,
        *,
        manifest_name: str = "embedding_manifest.json",
        vectors_name: str = "embeddings.npy",
    ) -> tuple[Path, Path]:
        # Save ``embeddings.npy`` plus a small manifest listing ``chunk_id`` per row.
        # Full chunk payloads are not duplicated here by default — keep your
        # ``chunks.json`` alongside for full traceability, or extend this dict.
        # Args:
        # vectors: ``(n, d)`` float32 matrix from :meth:`embed_texts`.
        # chunks: Same sequence used to build rows (for ``chunk_id`` alignment).
        # directory: Output folder (created if missing).
        # manifest_name: JSON sidecar filename.
        # vectors_name: NumPy matrix filename.
        # Returns:
        # Tuple ``(vectors_path, manifest_path)``.

        dir_path = ensure_dir(directory)
        vpath = dir_path / vectors_name
        mpath = dir_path / manifest_name

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {arr.shape}")
        np.save(vpath, arr)

        manifest: dict[str, Any] = {
            "model_name": self._model_name,
            "embedding_dim": int(arr.shape[1]),
            "num_vectors": int(arr.shape[0]),
            "rows": _chunks_to_manifest_rows(chunks),
        }
        save_json(manifest, mpath)
        return vpath.resolve(), mpath.resolve()

    @staticmethod
    def load_embeddings_matrix(path: str | Path) -> np.ndarray:
                # Load ``embeddings.npy`` saved by :meth:`save_embeddings_bundle`.

        p = Path(path)
        arr = np.load(p)
        return np.asarray(arr, dtype=np.float32)

    @staticmethod
    def load_manifest(path: str | Path) -> dict[str, Any]:
                # Load manifest JSON written next to embeddings.

        return load_json(path)
