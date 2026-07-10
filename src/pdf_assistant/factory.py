"""Wire together the real (production) implementations of the pipeline.

Isolated here so the API/UI don't import heavy dependencies directly and tests
can build a pipeline from fakes instead.
"""

from __future__ import annotations

from .config import Settings
from .embeddings import SentenceTransformerEmbedder
from .llm import AnthropicClient
from .pipeline import RAGPipeline
from .vector_store import ChromaVectorStore


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    """Construct a fully-wired production pipeline."""
    settings = settings or Settings.from_env()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = ChromaVectorStore(settings.persist_dir, settings.collection_name)
    llm = AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        max_tokens=settings.max_tokens,
    )
    return RAGPipeline(settings, embedder, store, llm)
