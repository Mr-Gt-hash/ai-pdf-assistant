"""Shared fixtures and lightweight fakes for the pipeline."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pytest

from pdf_assistant.config import Settings
from pdf_assistant.pipeline import RAGPipeline
from pdf_assistant.vector_store import Retrieved


class FakeEmbedder:
    """Deterministic, dependency-free embedder.

    Maps text -> a small fixed-length vector via hashing so identical text yields
    identical vectors and 'similarity' is meaningful enough for tests.
    """

    DIM = 8

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            h = hashlib.sha1(t.encode("utf-8")).digest()
            vectors.append([b / 255.0 for b in h[: self.DIM]])
        return vectors


class FakeVectorStore:
    """In-memory vector store: stores docs, returns them on query."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, ids, embeddings, documents, metadatas) -> None:
        for i, doc, meta in zip(ids, documents, metadatas):
            self.items.append({"id": i, "document": doc, "metadata": meta})

    def query(self, embedding, top_k, doc_id=None):
        items = [it for it in self.items if not doc_id or it["metadata"].get("doc_id") == doc_id]
        return [
            Retrieved(text=it["document"], metadata=it["metadata"], distance=0.0)
            for it in items[:top_k]
        ]

    def delete_document(self, doc_id) -> None:
        self.items = [it for it in self.items if it["metadata"].get("doc_id") != doc_id]


class FakeLLM:
    """Echoes the task so tests can assert prompt wiring without a real model."""

    def __init__(self) -> None:
        self.last_system = ""
        self.last_user = ""

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return f"ANSWER::{system[:20]}::{user[:20]}"


@pytest.fixture()
def settings() -> Settings:
    return Settings(anthropic_api_key="test-key", top_k=3, chunk_size=100, chunk_overlap=20)


@pytest.fixture()
def fakes():
    return FakeEmbedder(), FakeVectorStore(), FakeLLM()


@pytest.fixture()
def pipeline(settings, fakes):
    embedder, store, llm = fakes
    return RAGPipeline(settings, embedder, store, llm), embedder, store, llm
