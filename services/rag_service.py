"""
RAG Service — AI compliance recommendation pipeline.

Stage 1: generates dummy (mock) recommendations.
Stage 2: real retrieval (Chroma) + LLM generation.
"""

import random
import time

from storage.models import ReviewStatus, Judgment
from storage.repository import ReviewRepository, AuditRepository


class RAGService:

    @staticmethod
    def run_recommendation(request_id: str) -> list[dict]:
        """
        Run AI recommendation for every ReviewItem in the request.

        Stage 1: returns random mock judgments with dummy references.
        """
        detail = ReviewRepository.get_detail(request_id)
        if not detail:
            raise ValueError(f"Request {request_id} not found")

        # ── Transition: → AI_RUNNING ──
        ReviewRepository.update_request_status(
            request_id, ReviewStatus.AI_RUNNING.value
        )

        try:
            judgments = [
                Judgment.VIOLATION.value,
                Judgment.CAUTION.value,
                Judgment.OK.value,
            ]
            reason_map = {
                Judgment.VIOLATION.value: (
                    "해당 문구에 과대광고 또는 허위 표현 소지가 있습니다."
                ),
                Judgment.CAUTION.value: (
                    "해당 문구에 주의가 필요한 표현이 포함되어 있습니다."
                ),
                Judgment.OK.value: (
                    "해당 문구는 관련 규정상 문제가 없는 것으로 판단됩니다."
                ),
            }

            results = []
            for item in detail["items"]:
                start = time.time()
                time.sleep(0.05)  # simulate small latency
                latency = int((time.time() - start) * 1000)

                judgment = random.choice(judgments)
                text_preview = item["text"][:30]

                rec_id = ReviewRepository.create_ai_recommendation(
                    review_item_id=item["id"],
                    judgment=judgment,
                    reason=(
                        f"[단계1 더미] {reason_map[judgment]} "
                        f"문구: '{text_preview}...'"
                    ),
                    references=[
                        {
                            "doc_filename": "예시_법령.pdf",
                            "doc_type": "법령",
                            "page_or_row": "p.5",
                            "section_title": "제18조 (허위·과장 표시 금지)",
                            "relevance_score": round(
                                random.uniform(0.85, 0.99), 2
                            ),
                        },
                        {
                            "doc_filename": "방송심의지침.pdf",
                            "doc_type": "지침",
                            "page_or_row": "p.12",
                            "section_title": "3.2.1 효능효과 표현 기준",
                            "relevance_score": round(
                                random.uniform(0.75, 0.92), 2
                            ),
                        },
                        {
                            "doc_filename": "심의사례집.xlsx",
                            "doc_type": "사례",
                            "page_or_row": "Sheet1:Row23",
                            "section_title": "유사 사례 2024-A-0091",
                            "relevance_score": round(
                                random.uniform(0.60, 0.85), 2
                            ),
                        },
                    ],
                    model_name="dummy-mock-v1",
                    prompt_version="v0.1-stage1",
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
                    "model": "dummy-mock-v1",
                },
            )

            return results

        except Exception as e:
            # Rollback to REQUESTED so user can retry
            ReviewRepository.update_request_status(
                request_id, ReviewStatus.REQUESTED.value
            )
            raise e
