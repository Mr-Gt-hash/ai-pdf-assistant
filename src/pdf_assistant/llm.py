"""Anthropic LLM client wrapper.

Wraps the Anthropic Messages API. Imported lazily and defended behind our own
error type so the pipeline degrades gracefully when the key is missing.
"""

from __future__ import annotations

from typing import List, Protocol

from .errors import ConfigError, LLMError
from .logging_config import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    """Interface used by the pipeline (enables a fake in tests)."""

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Real client backed by the Anthropic API."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set.")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "anthropic is not installed. Run `pip install anthropic`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def complete(self, system: str, user: str) -> str:
        """Send a single-turn message and return the text response."""
        self._ensure_client()
        try:
            resp = self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        parts: List[str] = [
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ]
        text = "".join(parts).strip()
        if not text:
            raise LLMError("LLM returned an empty response.")
        return text
