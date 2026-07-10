"""Split extracted page text into overlapping chunks for embedding.

A character-based sliding window with overlap keeps chunks small enough for the
embedding model while preserving context across chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .extractor import Page
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A chunk of text plus provenance metadata."""

    text: str
    page: int
    index: int  # position of this chunk within the document


def chunk_pages(
    pages: List[Page], chunk_size: int = 1000, overlap: int = 150
) -> List[Chunk]:
    """Turn pages into overlapping chunks.

    Args:
        pages: Extracted pages.
        chunk_size: Max characters per chunk (must be > 0).
        overlap: Characters shared between consecutive chunks (0 <= overlap < chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    idx = 0
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page.number, index=idx))
                idx += 1
            start += step

    logger.info("Created %d chunk(s)", len(chunks))
    return chunks
