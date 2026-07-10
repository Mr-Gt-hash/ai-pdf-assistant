"""PDF text extraction.

Uses ``pypdf`` at runtime, imported lazily so unit tests that don't touch real
PDFs need not install it. Extraction returns per-page text so downstream code
can attach page numbers to chunks (useful for citations).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .errors import PDFExtractionError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Page:
    """A single extracted PDF page."""

    number: int   # 1-indexed
    text: str


def extract_pages(path: Path) -> List[Page]:
    """Extract text from each page of a PDF.

    Raises:
        PDFExtractionError: If the file is missing or cannot be parsed.
    """
    if not path.is_file():
        raise PDFExtractionError(f"PDF not found: {path}")

    try:
        from pypdf import PdfReader  # lazy import
    except ImportError as exc:  # pragma: no cover
        raise PDFExtractionError(
            "pypdf is not installed. Run `pip install pypdf`."
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises a variety of errors
        raise PDFExtractionError(f"Could not read PDF {path}: {exc}") from exc

    pages: List[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(number=i, text=text))

    if not pages:
        raise PDFExtractionError(f"No extractable text found in {path}.")

    logger.info("Extracted %d page(s) from %s", len(pages), path.name)
    return pages
