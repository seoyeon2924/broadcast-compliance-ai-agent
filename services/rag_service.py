"""
RAG Service — AI compliance recommendation pipeline.

Real retrieval (Chroma) + LLM generation via ReviewChain.
"""

import time

from langsmith import traceable

from chains.review_chain import ReviewChain
from storage.models import ReviewStatus
from storage.repository import ReviewRepository, AuditRepository


class RAGService:

    @staticmethod
    @traceable(name="run_recommendation", tags=["rag-service"])
    def run_recommendation(request_id: str) -> list[dict]:
        """
        Run AI recommendation for every ReviewItem in the request.

        Uses ReviewChain (Plan → Retrieve → Generate) for each item.
        """
        detail = ReviewRepository.get_detail(request_id)
        if not detail:
            raise ValueError(f"Request {request_id} not found")

        # ── Transition: → AI_RUNNING ──
        ReviewRepository.update_request_status(
            request_id, ReviewStatus.AI_RUNNING.value
        )

        try:
            chain = ReviewChain(model_name="gpt-4o-mini")

            results = []
            for item in detail["items"]:
                start = time.time()

                result = chain.run(
                    item_text=item["text"],
                    category=detail["request"]["category"],
                    broadcast_type=detail["request"]["broadcast_type"],
                )

                latency = int((time.time() - start) * 1000)

                rec_id = ReviewRepository.create_ai_recommendation(
                    review_item_id=item["id"],
                    judgment=result.get("judgment", "주의"),
                    reason=result.get("reason", ""),
                    references=result.get("references", []),
                    model_name=chain.model_name,
                    prompt_version="v1.0-rag-pipeline",
                    latency_ms=latency,
                )
                results.append({"item_id": item["id"], "rec_id": rec_id})

            # ── Transition: AI_RUNNING → REVIEWING ──
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REVIEWING.value
            )

            AuditRepository.create_log(
                event_type="AI_RECOMMEND",
                entity_type="ReviewRequest",
                entity_id=request_id,
                actor="system",
                detail={
                    "item_count": len(detail["items"]),
                    "model": chain.model_name,
                    "pipeline": "v1.0-rag",
                },
            )

            return results

        except Exception as e:
            # Rollback to REQUESTED so user can retry
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REQUESTED.value
            )
            raise e
