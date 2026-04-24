#
# Author: <YOUR_FULL_NAME>  |  Index: <YOUR_INDEX_NUMBER>
# ---------------------------------------------------------------------------
# llm_client.py — minimal, pluggable HTTP client for OpenAI-compatible chat APIs.
#
# Design:
#     - **No LangChain** — only stdlib ``urllib`` + ``json`` (no extra HTTP deps).
#     - Reads ``OPENAI_API_KEY`` from the environment (required for live calls).
#     - Optional ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``) so you can
#       point at proxies or other OpenAI-compatible servers later.
#     - Optional ``OPENAI_MODEL`` (default ``gpt-4o-mini``) for cheap class demos.
#
# Public API:
#     - :meth:`OpenAIChatClient.chat` — preferred for RAG (system + user messages).
#     - :meth:`OpenAIChatClient.generate` — convenience: single user message string
#       (useful for “pure LLM” baselines in :mod:`src.evaluation.compare_rag_vs_llm`).
#
# Each call returns a plain :class:`LLMResponse` dataclass suitable for logging.
#


from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class LLMResponse:
    # Structured result from the model for transparent logging.


    model: str
    output_text: str
    prompt_messages: list[dict[str, str]]
    # Echo of the message list sent to the API (for audit trails).

    latency_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    # Token usage when returned by the API (may be empty on errors).

    raw_response: dict[str, Any] | None = None
    # Parsed JSON body for deep debugging (optional).

    error: str | None = None
    # If set, ``output_text`` may be empty; ``raw_response`` may still hold detail.

    @property
    def prompt_for_log(self) -> str:
                # Single string echo of all messages (matches prompt_builder log style).

        parts: list[str] = []
        for m in self.prompt_messages:
            role = str(m.get("role", "")).upper()
            parts.append(f"===== {role} =====\n{m.get('content', '')}")
        return "\n\n".join(parts)

    def to_log_dict(self) -> dict[str, Any]:
                # Flat dict for :mod:`src.logging.logger` snapshots (JSON-serializable).

        return {
            "model": self.model,
            "prompt": self.prompt_for_log,
            "prompt_messages": list(self.prompt_messages),
            "output_text": self.output_text,
            "usage": dict(self.usage),
            "latency_seconds": self.latency_seconds,
            "error": self.error,
        }


class OpenAIChatClient:
    # Small OpenAI **Chat Completions** client (``/v1/chat/completions``).
    # Swap providers later by changing ``base_url`` or subclassing and overriding
    # :meth:`_post_json`.


    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self._timeout = float(timeout_seconds)

    @property
    def model(self) -> str:
        return self._model

    def _require_key(self) -> str:
        if not self._api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set it in your environment or pass api_key=..."
            )
        return self._api_key

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._require_key()}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling OpenAI: {exc}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from API: {raw[:500]}") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Call ``chat/completions`` with an explicit message list.
        # Args:
        # messages: OpenAI-format roles ``system`` / ``user`` / ``assistant``.
        # temperature: Sampling temperature (low for exam-style factual tasks).
        # max_tokens: Optional cap on completion tokens.
        # extra_body: Merge into the JSON body for small experiments (JSON mode, etc.).
        # Returns:
        # :class:`LLMResponse` with text, usage, latency, and optional ``raw_response``.

        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if extra_body:
            payload.update(extra_body)

        try:
            data = self._post_json("/chat/completions", payload)
        except Exception as exc:  # noqa: BLE001 — surfaced in LLMResponse for logging
            return LLMResponse(
                model=self._model,
                output_text="",
                prompt_messages=list(messages),
                latency_seconds=time.perf_counter() - t0,
                usage={},
                raw_response=None,
                error=str(exc),
            )

        text = self._extract_output_text(data)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return LLMResponse(
            model=str(data.get("model", self._model)),
            output_text=text,
            prompt_messages=list(messages),
            latency_seconds=time.perf_counter() - t0,
            usage=dict(usage),
            raw_response=data,
            error=None,
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Single-turn generation as **one user message** (no system role).
        # Intended for “pure LLM” comparisons where the entire prompt is one blob.

        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        ch0 = choices[0]
        if not isinstance(ch0, dict):
            return ""
        msg = ch0.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
        # legacy completions shape
        if isinstance(ch0.get("text"), str):
            return ch0["text"]
        return ""
