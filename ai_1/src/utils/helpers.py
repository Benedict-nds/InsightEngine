#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# helpers.py — small, explicit utilities used across the RAG pipeline.
#
# Responsibilities:
#     - Read/write JSON artifacts under ``data/processed`` or ``logs/``.
#     - Create output directories in a predictable way.
#     - Generate stable-enough identifiers for documents and chunks.
#     - Offer a tiny timing helper for demos and performance traces.
#
# This module intentionally stays dependency-free (stdlib only) so every
# other layer can import it without circular import issues.
#


from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


def ensure_dir(path: str | Path) -> Path:
    # Create a directory (and parents) if it does not exist.
    # Args:
    # path: Directory path as string or :class:`pathlib.Path`.
    # Returns:
    # The resolved :class:`pathlib.Path` for chaining.

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    # Serialize ``data`` to JSON using UTF-8 encoding.
    # Parent directories are created automatically.
    # Args:
    # data: Any JSON-serializable Python value.
    # path: Destination file path.
    # indent: Pretty-print indentation (default 2 for readability in exams).
    # Returns:
    # Resolved path written to disk.

    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return out.resolve()


def load_json(path: str | Path) -> Any:
    # Load JSON from disk.
    # Args:
    # path: Source file path.
    # Returns:
    # Parsed Python object (typically ``dict`` or ``list``).
    # Raises:
    # FileNotFoundError: If the file does not exist.
    # json.JSONDecodeError: If the file is not valid JSON.

    src = Path(path)
    with src.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_id(prefix: str | None = None) -> str:
    # Generate a unique identifier suitable for documents and chunks.
    # Uses UUID4 hex to avoid collisions in local experiments. Optional
    # ``prefix`` helps humans skim logs (e.g. ``doc_``, ``chunk_``).
    # Args:
    # prefix: Optional short label prepended with an underscore.
    # Returns:
    # Identifier string, e.g. ``doc_a1b2c3d4`` or raw hex.

    uid = uuid.uuid4().hex
    if prefix:
        return f"{prefix.rstrip('_')}_{uid}"
    return uid


@contextmanager
def timed(
    label: str,
    log_fn: Callable[[str], None] | None = None,
) -> Iterator[None]:
    # Context manager that measures elapsed wall time for a block.
    # Example::
    # with timed("load_pdf"):
    # extract_pdf_pages(path)
    # Args:
    # label: Human-readable label printed or passed to ``log_fn``.
    # log_fn: Optional callable (e.g. ``logger.info``) to receive one string.
    # Yields:
    # Nothing; used only for side effects (timing).

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        message = f"[timing] {label}: {elapsed_ms:.2f} ms"
        if log_fn is not None:
            log_fn(message)
        else:
            print(message)
