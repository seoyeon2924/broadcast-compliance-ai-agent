"""
방송 심의 AI Agent — MVP Entry Point

Run:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="방송 심의 AI Agent",
    page_icon="\U0001F3AC",
    layout="wide",
)

# ── Database initialisation ──
from storage.database import init_db  # noqa: E402

init_db()

# ── Session state defaults ──
if "current_page" not in st.session_state:
    st.session_state.current_page = "knowledge"

# ── Page registry ──
MAIN_PAGES: list[tuple[str, str]] = [
    ("\U0001F4DA 기준지식 관리", "knowledge"),
    ("\U0001F4DD 심의요청 등록", "request"),
    ("\U0001F4CB 심의요청 목록", "list"),
]

# ── Sidebar navigation ──
with st.sidebar:
    st.title("\U0001F3AC 방송 심의 AI Agent")
    st.caption("MVP v0.1 — 단계 1")
    st.divider()

    for label, page_key in MAIN_PAGES:
        is_current = st.session_state.current_page == page_key
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            if not is_current:
                st.session_state.current_page = page_key
                st.rerun()

    # Detail page indicator (navigated from list)
    if st.session_state.current_page == "detail":
        st.divider()
        st.caption("\U0001F50D 심의 상세 화면")
        if st.button(
            "\u2190 목록으로 돌아가기",
            key="nav_back",
            use_container_width=True,
        ):
            st.session_state.current_page = "list"
            st.rerun()

# ── Import renderers ──
from ui.page_knowledge import render as render_knowledge  # noqa: E402
from ui.page_request import render as render_request  # noqa: E402
from ui.page_list import render as render_list  # noqa: E402
from ui.page_review_detail import render as render_detail  # noqa: E402

RENDERERS: dict[str, callable] = {
    "knowledge": render_knowledge,
    "request": render_request,
    "list": render_list,
    "detail": render_detail,
}

# ── Render current page ──
RENDERERS[st.session_state.current_page]()
