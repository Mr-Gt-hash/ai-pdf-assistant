"""Embedding generation using Sentence Transformers.

The model is heavy to load, so it is created once and cached on the instance.
Import is lazy so tests can inject a fake embedder without the dependency.
"""

from __future__ import annotations

from typing import List, Protocol

from .errors import EmbeddingError
from .logging_config import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    """Minimal interface the rest of the app depends on (enables test fakes)."""

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class SentenceTransformerEmbedder:
    """Real embedder backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazily loaded on first use

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers`."
            ) from exc
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return an embedding vector for each input text."""
        if not texts:
            return []
        self._ensure_model()
        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                texts, convert_to_numpy=True, show_progress_bar=False
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return [v.tolist() for v in vectors]
