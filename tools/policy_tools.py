"""
규정·지침 검색 Tool — LangChain @tool 래퍼.

regulations(법률 + 규정) 컬렉션과 guidelines(지침) 컬렉션에서
관련 근거를 벡터 검색하여 반환한다.
"""

from __future__ import annotations

from langchain_core.tools import tool

from providers.embed_openai import OpenAIEmbedProvider
from storage.chroma_store import chroma_store


# ───────────────────────────────────────────
# 내부 헬퍼
# ───────────────────────────────────────────

_LAW_DOC_TYPES = {"법령", "law"}
_embedder: OpenAIEmbedProvider | None = None


def _get_query_embedding(query: str) -> list[float]:
    """인덱싱과 동일한 OpenAI 임베딩으로 쿼리 벡터 생성 (1536차원)."""
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbedProvider()
    return _embedder.embed([query])[0]


def _parse_query_result(raw: dict) -> list[dict]:
    """
    chroma_store.query() 반환값(2중 리스트)을 평탄화하여
    [{content, metadata, chroma_id, relevance_score}, ...] 리스트로 변환.
    """
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


# ───────────────────────────────────────────
# Tools
# ───────────────────────────────────────────

@tool
def search_policy(query: str) -> dict:
    """규정·지침 검색: 주어진 질의와 관련된 법률, 규정, 지침 근거를 벡터DB에서 검색합니다.

    Args:
        query: 검색할 질의 문자열 (예: "방송 한정판매 긴급성 표현")

    Returns:
        law_chunks — 법률 관련 청크 (regulations 컬렉션 중 doc_type이 법령/law)
        regulation_chunks — 규정 관련 청크 (regulations 컬렉션 중 나머지)
        guideline_chunks — 지침 관련 청크 (guidelines 컬렉션)
    """
    query_embedding = _get_query_embedding(query)

    # 1) regulations 컬렉션 검색
    reg_raw = chroma_store.query(
        collection_key="regulations",
        query_embeddings=[query_embedding],
        n_results=10,
    )
    reg_chunks = _parse_query_result(reg_raw)

    law_chunks: list[dict] = []
    regulation_chunks: list[dict] = []
    for chunk in reg_chunks:
        dt = chunk["metadata"].get("doc_type", "")
        if dt in _LAW_DOC_TYPES:
            law_chunks.append(chunk)
        else:
            regulation_chunks.append(chunk)

    # 2) guidelines 컬렉션 검색
    guide_raw = chroma_store.query(
        collection_key="guidelines",
        query_embeddings=[query_embedding],
        n_results=5,
    )
    guideline_chunks = _parse_query_result(guide_raw)

    return {
        "law_chunks": law_chunks,
        "regulation_chunks": regulation_chunks,
        "guideline_chunks": guideline_chunks,
    }


@tool
def fetch_chunk_by_id(chroma_id: str, collection_key: str) -> dict:
    """Chroma ID로 특정 청크 원문을 조회합니다.

    Args:
        chroma_id: 조회할 청크의 Chroma ID (예: "abc123_chunk_3")
        collection_key: 컬렉션 이름 (regulations / guidelines / cases / general)

    Returns:
        content — 청크 텍스트 내용
        metadata — 메타데이터
        chroma_id — 조회한 ID
    """
    coll = chroma_store.get_collection(collection_key)
    result = coll.get(ids=[chroma_id])

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        return {"content": "", "metadata": {}, "chroma_id": chroma_id}

    return {
        "content": documents[0] if documents else "",
        "metadata": metadatas[0] if metadatas else {},
        "chroma_id": ids[0],
    }
