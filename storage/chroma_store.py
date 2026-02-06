"""
ChromaDB persistent collection wrapper.

Stage 1: initialises the collection; real upsert/query used from Stage 2.
"""

import chromadb

from config import settings


class ChromaStore:
    """Thin wrapper around a single Chroma persistent collection."""

    COLLECTION_NAME = "compliance_chunks"

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ── Lazy properties ──

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR
            )
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Public API ──

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        kwargs: dict = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict = {"n_results": n_results}
        if query_texts:
            kwargs["query_texts"] = query_texts
        if query_embeddings:
            kwargs["query_embeddings"] = query_embeddings
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)


# Module-level singleton
chroma_store = ChromaStore()
