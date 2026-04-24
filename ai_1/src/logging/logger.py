#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# logger.py — JSON / JSONL logging for reproducible RAG experiments.
#
# Goals (exam / viva):
#     - One **session** per user query (or per notebook cell) with a stable ``session_id``.
#     - Append lightweight events to **JSONL** for long runs, and optionally write a
#       **pretty JSON snapshot** per session for slides.
#     - Capture retrieval scores, trimmed context, prompts, model output, latency, errors.
#
# This module does not configure Python's ``logging`` stdlib — names are separate to
# avoid confusion. All paths are under your ``logs/`` tree by default.
#


from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.utils.helpers import ensure_dir, generate_id, save_json


def utc_now_iso() -> str:
        # UTC timestamp with ``Z`` suffix (presentation-friendly).

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl(record: Mapping[str, Any], path: str | Path) -> Path:
    # Append one JSON object as a single line (JSONL).
    # Args:
    # record: Must be JSON-serializable.
    # path: Destination ``.jsonl`` file (parent dirs created on first write).
    # Returns:
    # Resolved path.

    p = Path(path)
    ensure_dir(p.parent)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
    return p.resolve()


def save_debug_snapshot(
    data: Any,
    directory: str | Path,
    *,
    name: str = "snapshot",
    prefix: str = "debug",
) -> Path:
    # Write a formatted JSON file: ``{prefix}_{utc}_{name}.json``.
    # Args:
    # data: Any JSON-serializable structure (dicts, lists, scalars).
    # directory: Folder under ``logs/`` (or absolute).
    # name: Short label (sanitized to filename-safe-ish).
    # prefix: File prefix.
    # Returns:
    # Path written.

    d = ensure_dir(directory)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:80]
    fname = f"{prefix}_{utc_now_iso().replace(':', '')}_{safe}.json"
    out = d / fname
    save_json(_to_jsonable(data), out)
    return out.resolve()


def _to_jsonable(obj: Any) -> Any:
        # Best-effort conversion for mixed dataclass / SDK objects in snapshots.

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def log_query_session(
    session_id: str,
    payload: Mapping[str, Any],
    *,
    jsonl_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    write_snapshot: bool = True,
) -> dict[str, Path | None]:
    # Log one end-to-end query session (single JSONL line + optional snapshot).
    # Typical ``payload`` keys (all optional but recommended):
    # - ``query``: user text
    # - ``retrieval_hits``: list of hit dicts (or counts only if large)
    # - ``prompt_metadata``: output from :func:`src.generation.prompt_builder.build_rag_prompt`
    # - ``llm_response``: dict from :class:`src.generation.llm_client.LLMResponse` fields
    # - ``errors``: list of strings
    # Args:
    # session_id: Correlates all records for one query.
    # payload: JSON-serializable session dict (nested dicts/lists OK).
    # jsonl_path: If set, append one line to this JSONL file.
    # snapshot_dir: If ``write_snapshot`` and set, write ``session_<id>.json`` here.
    # write_snapshot: When True and ``snapshot_dir`` set, save formatted JSON.
    # Returns:
    # Dict with keys ``jsonl`` and ``snapshot`` paths (``None`` if skipped).

    record = {
        "type": "query_session",
        "session_id": session_id,
        "ts": utc_now_iso(),
        **_to_jsonable(dict(payload)),
    }

    out_paths: dict[str, Path | None] = {"jsonl": None, "snapshot": None}

    if jsonl_path is not None:
        out_paths["jsonl"] = append_jsonl(record, jsonl_path)

    if write_snapshot and snapshot_dir is not None:
        snap_dir = ensure_dir(snapshot_dir)
        safe_sid = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:120]
        snap_path = snap_dir / f"session_{safe_sid}.json"
        save_json(record, snap_path)
        out_paths["snapshot"] = snap_path.resolve()

    return out_paths


class RagJsonLogger:
    # Small helper that fixes default paths under ``logs/queries`` and ``logs/experiments``.


    def __init__(
        self,
        *,
        queries_jsonl: str | Path | None = None,
        snapshots_dir: str | Path | None = None,
        experiments_jsonl: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self._queries_jsonl = Path(
            queries_jsonl or (root / "logs" / "queries" / "pipeline.jsonl")
        )
        self._snapshots_dir = Path(snapshots_dir or (root / "logs" / "queries" / "sessions"))
        self._experiments_jsonl = Path(
            experiments_jsonl or (root / "logs" / "experiments" / "runs.jsonl")
        )
        ensure_dir(self._queries_jsonl.parent)
        ensure_dir(self._snapshots_dir)
        ensure_dir(self._experiments_jsonl.parent)

    def new_session_id(self, prefix: str = "sess") -> str:
                # Generate a fresh session id (uuid-based).

        return generate_id(prefix)

    def log_pipeline_event(self, event_type: str, data: Mapping[str, Any]) -> Path:
                # Append a single tagged event (retrieval, prompt_built, llm_done, ...).

        rec = {"type": event_type, "ts": utc_now_iso(), **dict(data)}
        return append_jsonl(rec, self._queries_jsonl)

    def log_full_session(
        self,
        session_id: str,
        payload: Mapping[str, Any],
        *,
        write_snapshot: bool = True,
    ) -> dict[str, Path | None]:
                # JSONL append + optional per-session JSON snapshot.

        return log_query_session(
            session_id,
            payload,
            jsonl_path=self._queries_jsonl,
            snapshot_dir=self._snapshots_dir,
            write_snapshot=write_snapshot,
        )

    def log_experiment_run(self, experiment_name: str, metrics: Mapping[str, Any]) -> Path:
                # Append one experiment row (for adversarial / RAG-vs-LLM batches).

        rec = {
            "type": "experiment",
            "experiment": experiment_name,
            "ts": utc_now_iso(),
            **dict(metrics),
        }
        return append_jsonl(rec, self._experiments_jsonl)
