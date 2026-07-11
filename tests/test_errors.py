"""Verify the exception hierarchy is coherent."""

from __future__ import annotations

from pdf_assistant.errors import (
    ConfigError,
    DocumentNotFoundError,
    LLMError,
    PDFAssistantError,
    PDFExtractionError,
)


def test_all_inherit_base():
    for exc in [ConfigError, DocumentNotFoundError, LLMError, PDFExtractionError]:
        assert issubclass(exc, PDFAssistantError)
