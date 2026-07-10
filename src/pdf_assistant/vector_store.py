"""Vector database wrapper around ChromaDB.

Chroma is imported lazily. The wrapper exposes only the operations the app
needs (add, query, delete) so the storage backend could be swapped later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .errors import VectorStoreError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Retrieved:
    """A retrieved chunk with its similarity distance and metadata."""

    text: str
    metadata: Dict[str, Any]
    distance: float


class ChromaVectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            import chromadb  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "chromadb is not installed. Run `pip install chromadb`."
            ) from exc
        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Could not open vector store: {exc}") from exc

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert chunks into the collection."""
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to add vectors: {exc}") from exc

    def query(
        self, embedding: List[float], top_k: int, doc_id: str | None = None
    ) -> List[Retrieved]:
        """Return the ``top_k`` most similar chunks, optionally filtered by doc."""
        where = {"doc_id": doc_id} if doc_id else None
        try:
            res = self._collection.query(
                query_embeddings=[embedding], n_results=top_k, where=where
            )
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            Retrieved(text=d, metadata=m or {}, distance=float(dist))
            for d, m, dist in zip(docs, metas, dists)
        ]

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        try:
            self._collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc
