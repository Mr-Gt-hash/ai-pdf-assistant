"""RAG pipeline: ties extraction, chunking, embedding, storage and the LLM.

The pipeline depends on *interfaces* (Embedder, LLMClient, vector store) rather
than concrete classes, so tests can inject lightweight fakes and run the full
ingest/query flow without any heavy models or network calls.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from . import prompts
from .chunker import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .errors import DocumentNotFoundError
from .extractor import extract_pages
from .llm import LLMClient
from .logging_config import get_logger
from .vector_store import Retrieved

logger = get_logger(__name__)


def _doc_id(path: Path) -> str:
    """Stable id derived from file name + size (cheap, deterministic)."""
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


class RAGPipeline:
    """Orchestrates ingestion and querying of PDF documents."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store,  # ChromaVectorStore or compatible
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = vector_store
        self._llm = llm

    # --------------------------------------------------------------- ingest
    def ingest(self, path: Path) -> str:
        """Extract, chunk, embed and store a PDF. Returns its document id."""
        pages = extract_pages(path)
        chunks = chunk_pages(
            pages,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        doc_id = _doc_id(path)

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        ids = [f"{doc_id}:{c.index}" for c in chunks]
        metadatas = [
            {"doc_id": doc_id, "source": path.name, "page": c.page, "chunk": c.index}
            for c in chunks
        ]
        self._store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Ingested '%s' as doc_id=%s (%d chunks)", path.name, doc_id, len(chunks))
        return doc_id

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, query: str, doc_id: str | None = None) -> List[Retrieved]:
        """Embed the query and fetch the most relevant chunks."""
        vector = self._embedder.embed([query])[0]
        results = self._store.query(vector, top_k=self._settings.top_k, doc_id=doc_id)
        if not results:
            raise DocumentNotFoundError(
                "No relevant content found. Has a document been ingested?"
            )
        return results

    # ------------------------------------------------------------------ tasks
    def ask(self, question: str, doc_id: str | None = None) -> str:
        """Answer a question grounded in retrieved context."""
        chunks = self.retrieve(question, doc_id)
        context = prompts.build_context(chunks)
        return self._llm.complete(
            prompts.SYSTEM_QA, prompts.qa_user_prompt(question, context)
        )

    def _run_task(self, system: str, seed_query: str, doc_id: str | None) -> str:
        chunks = self.retrieve(seed_query, doc_id)
        context = prompts.build_context(chunks)
        return self._llm.complete(system, prompts.task_user_prompt(context))

    def summarize(self, doc_id: str | None = None) -> str:
        return self._run_task(prompts.SYSTEM_SUMMARY, "overview summary key points", doc_id)

    def extract_tables(self, doc_id: str | None = None) -> str:
        return self._run_task(prompts.SYSTEM_TABLES, "table figures rows columns data", doc_id)

    def find_dates(self, doc_id: str | None = None) -> str:
        return self._run_task(prompts.SYSTEM_DATES, "date deadline schedule year month", doc_id)

    def generate_report(self, doc_id: str | None = None) -> str:
        return self._run_task(prompts.SYSTEM_REPORT, "overview risks actions key points", doc_id)
