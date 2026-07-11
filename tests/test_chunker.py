"""Tests for chunking logic."""

from __future__ import annotations

import pytest

from pdf_assistant.chunker import Chunk, chunk_pages
from pdf_assistant.extractor import Page


def test_chunk_basic() -> None:
    pages = [Page(number=1, text="a" * 250)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].page == 1
    assert len(chunks[0].text) <= 100
    # step = 80, so starts at 0,80,160,240 -> 4 chunks
    assert len(chunks) == 4


def test_chunk_indices_are_sequential() -> None:
    pages = [Page(number=1, text="x" * 300), Page(number=2, text="y" * 90)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "abc")], chunk_size=10, overlap=10)


def test_chunk_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "abc")], chunk_size=0, overlap=0)
