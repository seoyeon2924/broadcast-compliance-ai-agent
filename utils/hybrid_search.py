"""
Hybrid Search — BM25 + Vector + Reciprocal Rank Fusion (RRF).

ChromaDB는 BM25를 네이티브 지원하지 않으므로,
별도 BM25 인덱스를 메모리에 구축하고 벡터 검색 결과와 RRF로 병합한다.

사용:
    from utils.hybrid_search import HybridSearchEngine
    engine = HybridSearchEngine()
    results = engine.search("cases", query_text, query_embedding, top_n=10)

아키텍처:
    1. 벡터 검색: ChromaDB cosine similarity (기존과 동일)
    2. BM25 검색: rank_bm25 라이브러리로 별도 인덱스
    3. RRF 병합: 두 결과의 순위를 Reciprocal Rank Fusion으로 결합
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from storage.chroma_store import chroma_store, ALL_COLLECTION_NAMES

logger = logging.getLogger(__name__)

# ── 한국어 간이 토크나이저 ─────────────────────────────────────────

_KO_SPLIT_RE = re.compile(r"[^\w가-힣]+")


def _tokenize_ko(text: str) -> list[str]:
    """한국어 + 영문 혼합 토크나이저 (공백/특수문자 분리 + 2자 이상)."""
    tokens = _KO_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) >= 2]


# ── BM25 인덱스 ────────────────────────────────────────────────────

class BM25Index:
    """단일 컬렉션용 BM25 인덱스."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._doc_metadatas: list[dict] = []
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)

    def build(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        """BM25 인덱스 구축."""
        self._doc_ids = ids
        self._doc_texts = documents
        self._doc_metadatas = metadatas

        tokenized = [_tokenize_ko(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._built = True
        logger.debug("BM25 인덱스 구축: %d건", len(ids))

    def search(self, query: str, top_n: int = 20) -> list[dict]:
        """BM25 검색 — 상위 top_n건 반환."""
        if not self._built or self._bm25 is None:
            return []

        tokens = _tokenize_ko(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # 상위 top_n 인덱스
        scored = sorted(enumerate(scores), key=lambda x: -x[1])[:top_n]

        results = []
        for idx, score in scored:
            if score <= 0:
                continue
            results.append({
                "content": self._doc_texts[idx],
                "metadata": self._doc_metadatas[idx],
                "chroma_id": self._doc_ids[idx],
                "bm25_score": round(float(score), 4),
            })
        return results


# ── RRF (Reciprocal Rank Fusion) ───────────────────────────────────

def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion으로 여러 검색 결과를 병합.

    각 결과 리스트에서 문서의 순위를 기반으로 점수를 계산:
        score(d) = Σ 1 / (k + rank_i(d))

    Args:
        *result_lists: 각각 [{chroma_id, content, metadata, ...}, ...] 형태
        k: RRF 상수 (기본값 60, 논문 권장)

    Returns:
        RRF 점수 기준 내림차순 정렬된 병합 결과
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc["chroma_id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            # 먼저 등장한 결과 리스트(벡터 검색)의 문서 정보를 우선 유지
            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()

    # RRF 점수 기준 정렬
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])

    merged = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        doc["relevance_score"] = doc.get("relevance_score", 0.0)
        merged.append(doc)

    return merged


# ── 하이브리드 검색 엔진 ───────────────────────────────────────────

class HybridSearchEngine:
    """BM25 + Vector + RRF 하이브리드 검색 엔진.

    Chroma 컬렉션별로 BM25 인덱스를 lazy-build하여 캐싱한다.
    첫 검색 시 해당 컬렉션의 전체 문서를 가져와 BM25 인덱스를 구축한다.
    """

    def __init__(self) -> None:
        self._bm25_indices: dict[str, BM25Index] = {}

    def _ensure_bm25_index(self, collection_key: str) -> BM25Index:
        """BM25 인덱스가 없으면 Chroma에서 전체 문서를 가져와 구축."""
        if collection_key in self._bm25_indices and self._bm25_indices[collection_key].is_built:
            return self._bm25_indices[collection_key]

        index = BM25Index()

        try:
            coll = chroma_store.get_collection(collection_key)
            total = coll.count()
            if total == 0:
                logger.warning("컬렉션 '%s'이 비어있음, BM25 인덱스 스킵", collection_key)
                self._bm25_indices[collection_key] = index
                return index

            # Chroma에서 전체 문서 로드 (배치)
            all_ids: list[str] = []
            all_docs: list[str] = []
            all_metas: list[dict] = []

            batch_size = 500
            offset = 0
            while offset < total:
                batch = coll.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                batch_ids = batch.get("ids", [])
                batch_docs = batch.get("documents", [])
                batch_metas = batch.get("metadatas", [])

                all_ids.extend(batch_ids)
                all_docs.extend(batch_docs)
                all_metas.extend(batch_metas)

                if len(batch_ids) < batch_size:
                    break
                offset += batch_size

            index.build(all_ids, all_docs, all_metas)
            logger.info(
                "BM25 인덱스 구축 완료: 컬렉션='%s', 문서=%d건",
                collection_key, index.doc_count,
            )

        except Exception as e:
            logger.error("BM25 인덱스 구축 실패 [%s]: %s", collection_key, e)

        self._bm25_indices[collection_key] = index
        return index

    def invalidate(self, collection_key: str | None = None) -> None:
        """BM25 인덱스 캐시 무효화 (재인덱싱 후 호출)."""
        if collection_key:
            self._bm25_indices.pop(collection_key, None)
        else:
            self._bm25_indices.clear()

    def search(
        self,
        collection_key: str,
        query_text: str,
        query_embedding: list[float],
        vector_top_n: int = 20,
        bm25_top_n: int = 20,
        final_top_n: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """하이브리드 검색 실행.

        Args:
            collection_key: Chroma 컬렉션 키 (cases, regulations, guidelines)
            query_text: 검색 쿼리 원문 (BM25용)
            query_embedding: 검색 쿼리 임베딩 벡터 (Vector용)
            vector_top_n: 벡터 검색 상위 N건
            bm25_top_n: BM25 검색 상위 N건
            final_top_n: RRF 병합 후 최종 반환 건수
            rrf_k: RRF 상수

        Returns:
            RRF 점수 기준 상위 final_top_n건
        """
        # 1. 벡터 검색 (기존 ChromaDB)
        try:
            raw = chroma_store.query(
                collection_key=collection_key,
                query_embeddings=[query_embedding],
                n_results=vector_top_n,
            )
            vector_results = _parse_chroma_result(raw)
        except Exception as e:
            logger.error("벡터 검색 실패 [%s]: %s", collection_key, e)
            vector_results = []

        # 2. BM25 검색
        bm25_index = self._ensure_bm25_index(collection_key)
        bm25_results = bm25_index.search(query_text, top_n=bm25_top_n)

        # 3. RRF 병합
        merged = reciprocal_rank_fusion(vector_results, bm25_results, k=rrf_k)

        logger.debug(
            "하이브리드 검색 [%s]: vector=%d + bm25=%d → rrf=%d → top_%d",
            collection_key,
            len(vector_results),
            len(bm25_results),
            len(merged),
            final_top_n,
        )

        return merged[:final_top_n]


def _parse_chroma_result(raw: dict) -> list[dict]:
    """Chroma query 결과를 표준 형태로 파싱."""
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


# ── 싱글톤 ─────────────────────────────────────────────────────────

_engine: HybridSearchEngine | None = None


def get_hybrid_engine() -> HybridSearchEngine:
    global _engine
    if _engine is None:
        _engine = HybridSearchEngine()
    return _engine
