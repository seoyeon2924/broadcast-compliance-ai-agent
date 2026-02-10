"""
ChromaDB collection wrapper.

문서 유형별 컬렉션 분리:
  regulations — 법률(law) + 규정(regulation)
  guidelines  — 지침(guideline)
  cases       — 과거 심의 사례(case)
  general     — fallback
"""

import chromadb

from config import settings

COLLECTION_REGULATIONS = "regulations"
COLLECTION_GUIDELINES = "guidelines"
COLLECTION_CASES = "cases"
COLLECTION_GENERAL = "general"

DOC_TYPE_TO_COLLECTION = {
    "law": COLLECTION_REGULATIONS,
    "regulation": COLLECTION_REGULATIONS,
    "규정": COLLECTION_REGULATIONS,
    "법령": COLLECTION_REGULATIONS,
    "guideline": COLLECTION_GUIDELINES,
    "지침": COLLECTION_GUIDELINES,
    "case": COLLECTION_CASES,
    "사례": COLLECTION_CASES,
}


def get_collection_name_for_doc_type(doc_type: str) -> str:
    return DOC_TYPE_TO_COLLECTION.get(doc_type, COLLECTION_GENERAL)


class ChromaStore:
    """문서 유형별 컬렉션 관리 + upsert/query."""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collections: dict[str, chromadb.Collection] = {}

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR
            )
        return self._client

    def get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        out = {}
        for k, v in meta.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v).strip()[:500]
        return out

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] | None = None,
        collection_key: str | None = None,
    ) -> None:
        name = collection_key or COLLECTION_GENERAL
        coll = self.get_collection(name)
        clean_metas = [self._sanitize_metadata(m) for m in metadatas]
        kwargs: dict = {
            "ids": ids,
            "documents": documents,
            "metadatas": clean_metas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        coll.upsert(**kwargs)

    def query(
        self,
        collection_key: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        name = collection_key or COLLECTION_GENERAL
        coll = self.get_collection(name)
        kwargs: dict = {"n_results": n_results}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        if where is not None:
            kwargs["where"] = where
        return coll.query(**kwargs)

    def count(self, collection_key: str | None = None) -> int:
        if collection_key:
            return self.get_collection(collection_key).count()
        total = 0
        for name in (COLLECTION_REGULATIONS, COLLECTION_GUIDELINES, COLLECTION_CASES, COLLECTION_GENERAL):
            try:
                total += self.client.get_collection(name).count()
            except Exception:
                pass
        return total

    def delete(self, ids: list[str], collection_key: str | None = None) -> None:
        name = collection_key or COLLECTION_GENERAL
        self.get_collection(name).delete(ids=ids)


chroma_store = ChromaStore()
