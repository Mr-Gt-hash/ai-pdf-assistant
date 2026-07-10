"""Application configuration, loaded from environment variables.

Keeping all tunables in one dataclass makes the app easy to configure via a
`.env` file or container environment, and easy to override in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the PDF assistant."""

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-4-8"
    max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1000          # characters per chunk
    chunk_overlap: int = 150        # overlap between consecutive chunks

    # Retrieval
    top_k: int = 4                  # number of chunks to retrieve per query

    # Vector store
    persist_dir: str = "./chroma_store"
    collection_name: str = "pdf_documents"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to defaults."""
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw and raw.isdigit() else default

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            max_tokens=_int("MAX_TOKENS", cls.max_tokens),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int("TOP_K", cls.top_k),
            persist_dir=os.getenv("PERSIST_DIR", cls.persist_dir),
            collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
        )
