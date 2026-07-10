"""Custom exception hierarchy for the PDF assistant.

A dedicated hierarchy lets the API layer map domain failures to clean HTTP
responses instead of leaking tracebacks.
"""

from __future__ import annotations


class PDFAssistantError(Exception):
    """Base class for all errors raised by this package."""


class PDFExtractionError(PDFAssistantError):
    """Raised when text cannot be extracted from a PDF."""


class DocumentNotFoundError(PDFAssistantError):
    """Raised when a query references a document that was never ingested."""


class EmbeddingError(PDFAssistantError):
    """Raised when the embedding model fails to encode text."""


class VectorStoreError(PDFAssistantError):
    """Raised when the vector database read/write fails."""


class LLMError(PDFAssistantError):
    """Raised when the LLM call fails or returns an unusable response."""


class ConfigError(PDFAssistantError):
    """Raised when required configuration (e.g. API key) is missing."""
