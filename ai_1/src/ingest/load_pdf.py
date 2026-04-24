#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# load_pdf.py — extract text from budget PDFs page by page using PyMuPDF.
#
# Responsibilities:
#     - Open a PDF and read each page as plain text.
#     - Attach stable metadata: source path, 1-based page index, document id.
#     - Optionally persist a JSON artifact under ``data/processed`` for traceability.
#
# Dependencies: ``pymupdf`` (import name ``fitz``).
#


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ingest.clean_data import clean_pdf_text
from src.utils.helpers import ensure_dir, generate_id, save_json


@dataclass
class PdfPageRecord:
    # One page of extracted PDF text with explicit metadata for RAG traceability.
    # Attributes:
    # doc_id: Unique id for this load operation (ties pages together).
    # source_path: Absolute or project-relative path to the PDF file.
    # page_number: 1-based page index (easier to explain in reports).
    # text: Raw extracted text for this page (cleaning applied separately if requested).
    # char_count: Length of ``text`` after optional cleaning (for logging).
    # metadata: Extra key-value pairs preserved through chunking.


    doc_id: str
    source_path: str
    page_number: int
    text: str
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.char_count == 0 and self.text is not None:
            object.__setattr__(self, "char_count", len(self.text))


def extract_pdf_pages(
    pdf_path: str | Path,
    *,
    doc_id: str | None = None,
    apply_cleaning: bool = False,
) -> list[PdfPageRecord]:
    # Extract text from every page of a PDF.
    # Args:
    # pdf_path: Path to the ``.pdf`` file on disk.
    # doc_id: Optional stable id; if omitted, a new id is generated with prefix ``pdf``.
    # apply_cleaning: If True, run :func:`src.ingest.clean_data.clean_pdf_text` per page.
    # Returns:
    # List of :class:`PdfPageRecord` ordered by ``page_number`` ascending.
    # Raises:
    # FileNotFoundError: If ``pdf_path`` does not exist.
    # RuntimeError: If PyMuPDF is not installed or the file cannot be opened.

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        import fitz  # type: ignore[import-untyped]  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from exc

    resolved_id = doc_id or generate_id("pdf")
    records: list[PdfPageRecord] = []

    try:
        doc = fitz.open(path)
    except Exception as exc:  # pragma: no cover - library-specific errors
        raise RuntimeError(f"Failed to open PDF: {path}") from exc

    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            raw_text = page.get_text("text") or ""
            text = clean_pdf_text(raw_text) if apply_cleaning else raw_text
            records.append(
                PdfPageRecord(
                    doc_id=resolved_id,
                    source_path=str(path),
                    page_number=i + 1,
                    text=text,
                    char_count=len(text),
                    metadata={"source_type": "pdf", "page_index_0": i},
                )
            )
    finally:
        doc.close()

    return records


def records_to_jsonable(records: list[PdfPageRecord]) -> list[dict[str, Any]]:
        # Convert dataclass records to plain dicts for JSON serialization.

    return [asdict(r) for r in records]


def load_pdf_document(
    pdf_path: str | Path,
    *,
    output_json: str | Path | None = None,
    doc_id: str | None = None,
    apply_cleaning: bool = True,
) -> list[PdfPageRecord]:
    # High-level loader: extract pages and optionally save processed JSON.
    # Args:
    # pdf_path: Input PDF path.
    # output_json: If set, write list of page dicts to this path (UTF-8 JSON).
    # doc_id: Optional document id shared by all pages.
    # apply_cleaning: Whether to run PDF cleaning (recommended for RAG).
    # Returns:
    # Same as :func:`extract_pdf_pages`.

    pages = extract_pdf_pages(
        pdf_path, doc_id=doc_id, apply_cleaning=apply_cleaning
    )
    if output_json is not None:
        out_path = Path(output_json)
        ensure_dir(out_path.parent)
        save_json(records_to_jsonable(pages), out_path)
    return pages
