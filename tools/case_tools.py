"""
심의 사례 검색 Tool — LangChain @tool 래퍼.

cases 컬렉션에서 과거 심의 사례를 벡터 검색 → Cohere 리랭킹하여 반환한다.
"""

from __future__ import annotations

from langchain_core.tools import tool

from providers.embed_openai import OpenAIEmbedProvider
from storage.chroma_store import chroma_store
from utils.reranker import rerank_chunks  # 🆕 리랭커 추가

_embedder: OpenAIEmbedProvider | None = None


def _get_query_embedding(query: str) -> list[float]:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedProvider()
    return _embedder.embed([query])[0]


def _parse_query_result(raw: dict) -> list[dict]:
    ids = raw.get("ids", [[]])[0] or []
    documents = raw.get("documents", [[]])[0] or []
    metadatas = raw.get("metadatas", [[]])[0] or []
    distances = raw.get("distances", [[]])[0] or []

    chunks: list[dict] = []
    for idx in range(len(ids)):
        distance = distances[idx] if idx < len(distances) else 0.0
        chunks.append({
            "content": documents[idx] if idx < len(documents) else "",
            "metadata": metadatas[idx] if idx < len(metadatas) else {},
            "chroma_id": ids[idx],
            "relevance_score": round(1.0 - distance, 4),
        })
    return chunks


@tool
def search_cases(query: str) -> dict:
    """심의 사례 검색: 주어진 질의와 유사한 과거 심의 사례를 벡터DB에서 검색합니다.

    Args:
        query: 검색할 질의 문자열 (예: "오늘만 한정 가격")

    Returns:
        case_chunks — 유사 심의 사례 청크 목록 (최대 5건)
    """
    query_embedding = _get_query_embedding(query)

    # 1단계: ChromaDB에서 넉넉하게 검색 (기존 5 → 20)
    raw = chroma_store.query(
        collection_key="cases",
        query_embeddings=[query_embedding],
        n_results=20,  # 🆕 리랭킹용으로 넉넉하게
    )
    case_chunks = _parse_query_result(raw)

    # 2단계: Cohere 리랭킹으로 정밀 재정렬
    reranked = rerank_chunks(query=query, chunks=case_chunks, top_n=5)

    return {"case_chunks": reranked}