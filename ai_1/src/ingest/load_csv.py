#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# load_csv.py — load election (or tabular) CSV data as row-level "documents".
#
# Responsibilities:
#     - Load CSV with pandas using UTF-8 (with latin-1 fallback for messy files).
#     - Normalize column names to snake_case for stable keys.
#     - Convert each row into a text blob + metadata suitable for chunking and FAISS.
#
# No LangChain: only pandas + explicit string formatting.
#


from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.helpers import generate_id, save_json


def normalize_column_names(columns: list[str]) -> list[str]:
    # Normalize header strings to lowercase snake_case.
    # Rules:
    # - Strip leading/trailing spaces
    # - Lowercase
    # - Replace non-alphanumeric runs with single underscore
    # - Collapse duplicate underscores and trim edges
    # Args:
    # columns: Raw column names from the CSV header row.
    # Returns:
    # New list of normalized names (same length and order).

    import re

    out: list[str] = []
    for col in columns:
        s = str(col).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        out.append(s or "column")
    return _make_unique(out)


def _make_unique(names: list[str]) -> list[str]:
        # Ensure column names are unique by appending _2, _3, ... if needed.

    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
    return result


def row_to_text(row: pd.Series, columns: list[str]) -> str:
    # Build a single human-readable string from one CSV row.
    # Format: ``"column: value | column: value"`` so retrieval stays interpretable.
    # Args:
    # row: One row of the DataFrame.
    # columns: Column names to include (already normalized).
    # Returns:
    # Flattened string representation of the row.

    parts: list[str] = []
    for col in columns:
        val = row.get(col, "")
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val).strip()
        if val_str:
            parts.append(f"{col}: {val_str}")
    return " | ".join(parts)


@dataclass
class CsvRowDocument:
    # One CSV row packaged as a document for the RAG pipeline.
    # Attributes:
    # doc_id: Unique id for this row document.
    # source_path: Path to the CSV file.
    # row_index: 0-based row position in the file after header (pandas iloc).
    # text: Flattened row string for embedding/chunking.
    # fields: Dict of column -> raw cell value (JSON-serializable where possible).
    # metadata: Extra metadata (e.g. election year column if present).


    doc_id: str
    source_path: str
    row_index: int
    text: str
    fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_csv_as_documents(
    csv_path: str | Path,
    *,
    doc_id_prefix: str = "csv",
    encoding: str | None = None,
) -> tuple[pd.DataFrame, list[CsvRowDocument]]:
    # Load a CSV file and return both the DataFrame and row-level documents.
    # Args:
    # csv_path: Path to ``.csv``.
    # doc_id_prefix: Prefix for generated per-row ids.
    # encoding: Optional encoding; if None, tries utf-8 then latin-1.
    # Returns:
    # Tuple of ``(dataframe, list_of_csv_row_documents)``.
    # Raises:
    # FileNotFoundError: If path does not exist.
    # ValueError: If the file cannot be parsed as CSV.

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    if encoding is not None:
        df = pd.read_csv(path, encoding=encoding)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    df.columns = normalize_column_names(list(df.columns.astype(str)))

    documents: list[CsvRowDocument] = []
    for i in range(len(df)):
        row = df.iloc[i]
        fields = {col: _json_safe_scalar(row[col]) for col in df.columns}
        text = row_to_text(row, list(df.columns))
        doc = CsvRowDocument(
            doc_id=generate_id(doc_id_prefix),
            source_path=str(path),
            row_index=i,
            text=text,
            fields=fields,
            metadata={
                "source_type": "csv",
                "row_index": i,
            },
        )
        documents.append(doc)

    return df, documents


def _json_safe_scalar(val: Any) -> Any:
        # Convert pandas/NaN scalars to JSON-friendly Python types.

    if pd.isna(val):
        return None
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            return str(val)
    return val


def documents_to_jsonable(docs: list[CsvRowDocument]) -> list[dict[str, Any]]:
        # Serialize row documents to plain dicts.

    return [asdict(d) for d in docs]


def load_csv_and_save_processed(
    csv_path: str | Path,
    output_json: str | Path,
    *,
    encoding: str | None = None,
) -> list[CsvRowDocument]:
    # Load CSV and write row documents to JSON for reproducibility.
    # Args:
    # csv_path: Input CSV.
    # output_json: Output path for list of :class:`CsvRowDocument` dicts.
    # encoding: Optional file encoding.
    # Returns:
    # List of row documents (same content written to disk).

    _, docs = load_csv_as_documents(csv_path, encoding=encoding)
    save_json(documents_to_jsonable(docs), output_json)
    return docs
