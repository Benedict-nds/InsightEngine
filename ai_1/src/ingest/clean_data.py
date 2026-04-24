#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# clean_data.py — deterministic text normalization for PDFs and tabular text.
#
# Responsibilities:
#     - Collapse noisy whitespace left by PDF extractors.
#     - Repair common PDF line-break artifacts (hyphenation, lone newlines).
#     - Strip repeated boilerplate lines when they appear as obvious artifacts.
#
# Design goal: every transformation is easy to justify in an exam viva — no
# hidden ML, only explicit string rules.
#


from __future__ import annotations

import re
from collections.abc import Iterable


def normalize_whitespace(text: str) -> str:
    # Collapse runs of whitespace to single spaces and trim ends.
    # Args:
    # text: Raw string from PDF or CSV.
    # Returns:
    # Normalized string.

    if not text:
        return ""
    # Replace any whitespace run (including \\r, \\n, tabs) with one space
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.strip()


def fix_hyphen_line_breaks(text: str) -> str:
    # Join words split across lines with a hyphen (common PDF pattern).
    # Example: ``"exam-\nple"`` → ``"example"``.
    # Args:
    # text: Text possibly containing hyphen + newline splits.
    # Returns:
    # Text with those breaks repaired where the pattern matches.

    if not text:
        return ""
    # Hyphen at end of line followed by newline and continuation
    return re.sub(r"-\s*\n\s*", "", text)


def fix_broken_line_breaks(text: str) -> str:
    # Replace single newlines inside paragraphs with spaces.
    # Multi-paragraph blocks separated by blank lines are preserved in a
    # simple way: double newlines become a paragraph marker placeholder,
    # single newlines become spaces, then markers become double newlines.
    # Args:
    # text: Raw multi-line string.
    # Returns:
    # Text with softer line breaks merged for reading continuity.

    if not text:
        return ""
    # Preserve explicit paragraph gaps
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    parts = text.split("\n\n")
    merged_paragraphs: list[str] = []
    for block in parts:
        inner = re.sub(r"(?<!\n)\n(?!\n)", " ", block.strip())
        merged_paragraphs.append(inner)
    return "\n\n".join(p for p in merged_paragraphs if p)


def remove_repeated_lines(
    text: str,
    *,
    min_repeats: int = 3,
    min_line_length: int = 12,
) -> str:
    # Drop lines that repeat many times in a row (typical header/footer noise).
    # This is conservative: only lines at least ``min_line_length`` chars
    # and repeated at least ``min_repeats`` consecutive times are removed.
    # Args:
    # text: Full page or document text with newlines.
    # min_repeats: Consecutive identical lines required to treat as artifact.
    # min_line_length: Ignore very short repeated tokens.
    # Returns:
    # Text with obvious repeated boilerplate lines removed.

    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if len(line.strip()) < min_line_length:
            out.append(line)
            i += 1
            continue
        run = 1
        j = i + 1
        while j < len(lines) and lines[j] == line:
            run += 1
            j += 1
        if run >= min_repeats:
            # Skip this repeated boilerplate block
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def clean_general_text(text: str) -> str:
    # Cleaning path for CSV-derived or other plain text (no PDF hyphen pass).
    # Order: hyphen-newline repair → soft line merges → whitespace normalize.
    # Args:
    # text: Raw string from a table cell or concatenated row fields.
    # Returns:
    # Cleaned string safe for chunking and display.

    if not text:
        return ""
    t = fix_hyphen_line_breaks(text)
    t = fix_broken_line_breaks(t)
    return normalize_whitespace(t)


def clean_pdf_text(text: str) -> str:
    # Full PDF-oriented cleaning pipeline.
    # Steps:
    # 1. Fix hyphenated line breaks.
    # 2. Merge soft line breaks inside paragraphs.
    # 3. Remove long runs of identical lines (headers/footers).
    # 4. Normalize whitespace.
    # Args:
    # text: Raw page or document text from a PDF extractor.
    # Returns:
    # Cleaned text ready for chunking.

    if not text:
        return ""
    t = fix_hyphen_line_breaks(text)
    t = fix_broken_line_breaks(t)
    t = remove_repeated_lines(t)
    return normalize_whitespace(t)


def clean_batch(strings: Iterable[str], *, mode: str = "pdf") -> list[str]:
    # Apply the same cleaner to many strings (convenience for maps).
    # Args:
    # strings: Iterable of raw strings.
    # mode: ``"pdf"`` uses :func:`clean_pdf_text`; anything else uses
    # :func:`clean_general_text`.
    # Returns:
    # List of cleaned strings in the same order.

    cleaner = clean_pdf_text if mode == "pdf" else clean_general_text
    return [cleaner(s) for s in strings]
