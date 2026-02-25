"""
Page 4 — 심의 상세 (심의자)
AI recommendation + human final decision in a single view.
"""

import streamlit as st

from services.review_service import ReviewService
from services.rag_service import RAGService
from ui.components.status_badge import render_status_badge


def render() -> None:
    st.header("심의 상세")

    request_id = st.session_state.get("selected_request_id")
    if not request_id:
        st.warning(
            "심의 요청을 먼저 선택해주세요. "
            "[심의요청 목록]에서 '보기'를 클릭하세요."
        )
        return

    detail = ReviewService.get_detail(request_id)
    if not detail:
        st.error("요청을 찾을 수 없습니다.")
        return

    req = detail["request"]
    items = detail["items"]
    human_dec = detail["human_decision"]

    # ──────────────────────────────────
    # 1. Request Summary
    # ──────────────────────────────────
    st.subheader("요청 요약")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("상품명", req["product_name"])
    mc2.metric("카테고리", req["category"] or "-")
    mc3.metric("방송유형", req["broadcast_type"] or "-")
    with mc4:
        st.markdown("**상태**")
        render_status_badge(req["status"])

    created_str = (
        req["created_at"].strftime("%Y-%m-%d %H:%M")
        if req["created_at"]
        else "-"
    )
    st.caption(
        f"요청자: {req['requested_by'] or '-'} · "
        f"요청일: {created_str} · "
        f"문구 수: {len(items)}"
    )

    st.divider()

    # ──────────────────────────────────
    # 2. AI Recommendation
    # ──────────────────────────────────
    st.subheader("AI 심의 추천")

    can_run_ai = req["status"] in ("REQUESTED",)
    if st.button(
        "AI 심의 추천 실행",
        type="primary",
        disabled=not can_run_ai,
    ):
        with st.spinner("AI 추천 생성 중..."):
            try:
                RAGService.run_recommendation(request_id)
                st.success("AI 추천이 완료되었습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"AI 추천 실패: {e}")

    if req["status"] == "AI_RUNNING":
        st.info("AI가 추천 결과를 생성 중입니다...")

    # ── Item tabs ──
    if items:
        tab_labels = [item["label"] for item in items]
        tabs = st.tabs(tab_labels)

        for tab, item in zip(tabs, items):
            with tab:
                type_label = (
                    "요청문구" if item["item_type"] == "REQUEST_TEXT"
                    else "강조바"
                )
                st.markdown(
                    f"**{item['label']}** · `{type_label}`"
                )
                st.info(item["text"])

                rec = item.get("ai_recommendation")
                if rec:
                    _render_recommendation(rec)
                else:
                    st.caption(
                        "아직 AI 추천이 실행되지 않았습니다."
                    )

    st.divider()

    # ──────────────────────────────────
    # 3. Human Decision
    # ──────────────────────────────────
    st.subheader("최종 심의 판단")

    if human_dec:
        icon = "완료" if human_dec["decision"] == "DONE" else "반려"
        st.success(f"최종 결정: {icon}")
        st.markdown(f"**코멘트:** {human_dec['comment'] or '-'}")
        dec_date = (
            human_dec["created_at"].strftime("%Y-%m-%d %H:%M")
            if human_dec["created_at"]
            else "-"
        )
        st.caption(
            f"심의자: {human_dec['decided_by'] or '-'} · "
            f"결정일: {dec_date}"
        )
    else:
        can_decide = req["status"] == "REVIEWING"

        if not can_decide:
            st.info("AI 추천 실행 후 최종 판단을 내릴 수 있습니다.")

        with st.form("decision_form"):
            decision = st.radio(
                "최종 결과",
                ["DONE", "REJECTED"],
                format_func=lambda x: (
                    "완료 (DONE)" if x == "DONE" else "반려 (REJECTED)"
                ),
                horizontal=True,
            )
            comment = st.text_area(
                "심의 코멘트", placeholder="심의 의견을 작성하세요"
            )
            decided_by = st.text_input(
                "심의자", placeholder="예: 박심의위원"
            )

            submitted = st.form_submit_button(
                "최종 판단 저장",
                type="primary",
                disabled=not can_decide,
            )

        if submitted and can_decide:
            try:
                ReviewService.submit_decision(
                    request_id=request_id,
                    decision=decision,
                    comment=comment,
                    decided_by=decided_by,
                )
                label = "완료" if decision == "DONE" else "반려"
                st.success(f"최종 판단이 저장되었습니다: {label}")
                st.rerun()
            except Exception as e:
                st.error(f"저장 실패: {e}")


# ──────────────────────────────────
# Helper
# ──────────────────────────────────

_JUDGMENT_ICON = {
    "위반소지": "\U0001F534",
    "주의": "\U0001F7E1",
    "OK": "\U0001F7E2",
}


def _render_recommendation(rec: dict) -> None:
    """Render a single AI recommendation block."""
    icon = _JUDGMENT_ICON.get(rec["judgment"], "\u26AA")
    st.markdown(f"**판단:** {icon} {rec['judgment']}")
    st.markdown(f"**사유:** {rec['reason']}")

    all_refs = rec.get("references") or []
    refs = [r for r in all_refs if (r.get("content") or "").strip()]

    if refs:
        st.markdown(f"**근거:** ({len(refs)}건)")
        for i, ref in enumerate(refs, 1):
            doc_type = ref.get("doc_type", "-")
            case_number = ref.get("case_number", "")
            case_date = ref.get("case_date", "")
            article_number = ref.get("article_number", "")
            doc_filename = ref.get("doc_filename", "-")
            section_title = ref.get("section_title", "")
            score = ref.get("relevance_score", "-")
            content = ref.get("content", "")

            if doc_type == "사례" and case_number:
                date_str = f" ({case_date})" if case_date else ""
                label = f"처리번호 {case_number}{date_str}"
            elif article_number:
                label = f"`{doc_filename}` {article_number}"
                if section_title:
                    label += f" ({section_title})"
            else:
                label = f"`{doc_filename}`"
                if section_title:
                    label += f" · {section_title}"

            expander_title = f"{i}. [{doc_type}] {label} · score: {score}"
            with st.expander(expander_title):
                st.caption(content)
    elif all_refs:
        st.caption("⚠️ 근거 문서를 검색했으나 유효한 내용을 가져오지 못했습니다.")

    st.caption(
        f"모델: {rec.get('model_name', '-')} · "
        f"프롬프트: {rec.get('prompt_version', '-')} · "
        f"지연: {rec.get('latency_ms', '-')}ms"
    )
