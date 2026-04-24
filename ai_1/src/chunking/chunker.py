#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# chunker.py — split long text into overlapping windows and map CSV rows to chunks.
#
# Responsibilities:
#     - ``chunk_text``: fixed character window with explicit overlap (exam-friendly).
#     - ``document_to_chunks``: attach chunk_id, parent_doc_id, offsets, source metadata.
#     - Support PDF page records and CSV row documents via a small unified input shape.
#
# Chunking justification (for your report):
#     - Character windows are simple to explain and reproduce across models.
#     - Overlap reduces boundary effects where sentences span two chunks.
#     - CSV rows are usually one chunk each; very long rows are split with the same
#       ``chunk_text`` logic so the code path stays unified.
#
# This file does not call embedding or FAISS — it only produces chunk records.
#


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.ingest.clean_data import clean_general_text, clean_pdf_text
from src.ingest.load_csv import CsvRowDocument
from src.ingest.load_pdf import PdfPageRecord
from src.utils.helpers import generate_id, save_json


SourceKind = Literal["pdf_page", "csv_row", "text"]


@dataclass
class ChunkRecord:
    # One chunk ready for embedding and FAISS upsert.
    # Attributes:
    # chunk_id: Globally unique chunk identifier.
    # parent_doc_id: Id of the originating document (PDF load or CSV row).
    # text: Chunk text passed to the embedder.
    # source_kind: ``pdf_page``, ``csv_row``, or ``text``.
    # chunk_index: 0-based index among chunks from the same parent.
    # char_start: Start offset in parent text (inclusive) for pdf/text splits.
    # char_end: End offset in parent text (exclusive).
    # metadata: Copied/merged metadata for logging and UI (page, path, etc.).


    chunk_id: str
    parent_doc_id: str
    text: str
    source_kind: SourceKind
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    # Split ``text`` into overlapping character windows.
    # Args:
    # text: Full string to segment (already cleaned upstream if desired).
    # chunk_size: Maximum characters per chunk (must be >= 1).
    # overlap: Characters shared with the previous window (must be < chunk_size).
    # Returns:
    # List of ``(char_start, char_end, chunk_text)`` tuples. ``char_end`` is
    # exclusive Python slice style.
    # Raises:
    # ValueError: If parameters are inconsistent.

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    if not text:
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        chunks.append((start, end, piece))
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
        # Ensure forward progress if overlap would stall on tiny texts
        if start + overlap >= end and end < n:
            start = end
    return chunks


def _merge_metadata(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(extra)
    return merged


def pdf_pages_to_chunks(
    pages: list[PdfPageRecord],
    *,
    chunk_size: int = 900,
    overlap: int = 120,
    apply_cleaning: bool = False,
) -> list[ChunkRecord]:
    # Turn PDF page records into chunk records (one page may yield many chunks).
    # Args:
    # pages: Output from :func:`src.ingest.load_pdf.extract_pdf_pages`.
    # chunk_size: Character window for long pages.
    # overlap: Overlap between consecutive windows on the same page.
    # apply_cleaning: If True, run :func:`clean_pdf_text` before windowing
    # (use False if pages were already cleaned at extract time).
    # Returns:
    # Ordered list of :class:`ChunkRecord`.

    out: list[ChunkRecord] = []
    for page in pages:
        body = clean_pdf_text(page.text) if apply_cleaning else page.text
        triples = chunk_text(body, chunk_size, overlap)
        if not triples:
            continue
        for idx, (cs, ce, piece) in enumerate(triples):
            meta = _merge_metadata(
                page.metadata,
                {
                    "source_path": page.source_path,
                    "page_number": page.page_number,
                    "total_chunks_for_parent": len(triples),
                },
            )
            out.append(
                ChunkRecord(
                    chunk_id=generate_id("chunk"),
                    parent_doc_id=page.doc_id,
                    text=piece,
                    source_kind="pdf_page",
                    chunk_index=idx,
                    char_start=cs,
                    char_end=ce,
                    metadata=meta,
                )
            )
    return out


def csv_rows_to_chunks(
    rows: list[CsvRowDocument],
    *,
    chunk_size: int = 900,
    overlap: int = 120,
    apply_cleaning: bool = True,
) -> list[ChunkRecord]:
    # Convert CSV row documents to chunks (typically one chunk per row).
    # Very long rows are split with :func:`chunk_text` using the same parameters
    # as PDF pages so hybrid retrieval sees a uniform chunk size distribution.
    # Args:
    # rows: List of :class:`CsvRowDocument`.
    # chunk_size: Max characters per chunk when a row must be split.
    # overlap: Overlap for split rows.
    # apply_cleaning: If True, run :func:`clean_general_text` on row text first.
    # Returns:
    # List of :class:`ChunkRecord`.

    out: list[ChunkRecord] = []
    for row in rows:
        body = clean_general_text(row.text) if apply_cleaning else row.text
        triples = chunk_text(body, chunk_size, overlap)
        if not triples:
            continue
        for idx, (cs, ce, piece) in enumerate(triples):
            meta = _merge_metadata(
                row.metadata,
                {
                    "source_path": row.source_path,
                    "csv_row_index": row.row_index,
                    "fields": row.fields,
                    "total_chunks_for_parent": len(triples),
                },
            )
            out.append(
                ChunkRecord(
                    chunk_id=generate_id("chunk"),
                    parent_doc_id=row.doc_id,
                    text=piece,
                    source_kind="csv_row",
                    chunk_index=idx,
                    char_start=cs,
                    char_end=ce,
                    metadata=meta,
                )
            )
    return out


def plain_text_to_chunks(
    text: str,
    *,
    parent_doc_id: str | None = None,
    source_path: str = "",
    chunk_size: int = 900,
    overlap: int = 120,
    apply_cleaning: bool = True,
) -> list[ChunkRecord]:
    # Chunk arbitrary long text (useful for tests or future sources).
    # Args:
    # text: Raw or semi-structured text.
    # parent_doc_id: Optional id; generated if omitted.
    # source_path: Optional path string stored in metadata.
    # chunk_size: Window size.
    # overlap: Overlap size.
    # apply_cleaning: Use general cleaner when True.
    # Returns:
    # List of :class:`ChunkRecord` with ``source_kind="text"``.

    pid = parent_doc_id or generate_id("textdoc")
    body = clean_general_text(text) if apply_cleaning else text
    triples = chunk_text(body, chunk_size, overlap)
    out: list[ChunkRecord] = []
    for idx, (cs, ce, piece) in enumerate(triples):
        out.append(
            ChunkRecord(
                chunk_id=generate_id("chunk"),
                parent_doc_id=pid,
                text=piece,
                source_kind="text",
                chunk_index=idx,
                char_start=cs,
                char_end=ce,
                metadata={
                    "source_path": source_path,
                    "total_chunks_for_parent": len(triples),
                },
            )
        )
    return out


def chunks_to_jsonable(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
        # Serialize chunk records for ``chunks.json`` artifacts.

    return [asdict(c) for c in chunks]


def save_chunks_json(chunks: list[ChunkRecord], path: str) -> None:
        # Write chunks to JSON (UTF-8) using shared helper.

    save_json(chunks_to_jsonable(chunks), path)
