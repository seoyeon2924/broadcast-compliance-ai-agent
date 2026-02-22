"""
ReviewChain — LangGraph 기반 Self-Corrective RAG (Reflexion 아키텍처).

흐름: plan → retrieve → grade_documents → (rewrite_query 루프) → generate → grade_answer → (루프) → end
"""

from __future__ import annotations

import json
import logging
import operator
import time
from typing import Annotated, TypedDict

import httpx
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import settings
from prompts.generator import GENERATOR_HUMAN, GENERATOR_SYSTEM
from prompts.grader import (
    GRADE_ANSWER_HUMAN,
    GRADE_ANSWER_SYSTEM,
    REWRITE_QUERY_HUMAN,
    REWRITE_QUERY_SYSTEM,
)
from prompts.planner import PLANNER_HUMAN, PLANNER_SYSTEM
from tools.case_tools import search_cases
from tools.policy_tools import search_policy

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────
# State
# ───────────────────────────────────────────

class ReviewState(TypedDict):
    """LangGraph 상태 정의. 모든 노드가 이 State를 공유한다."""

    item_text: str
    category: str
    broadcast_type: str
    plan: dict
    context: dict
    relevant_doc_count: int
    result: dict
    answer_grade: str
    retry_count: int
    max_retries: int
    tool_logs: Annotated[list, operator.add]


# ───────────────────────────────────────────
# LLM / Parser (모듈 레벨)
# ───────────────────────────────────────────

_http_client = httpx.Client(verify=False)


def _get_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=settings.OPENAI_API_KEY or "dummy",
        http_client=_http_client,
        request_timeout=90,
    )


_llm = _get_llm()
_json_parser = JsonOutputParser()


# ───────────────────────────────────────────
# 헬퍼
# ───────────────────────────────────────────

def _format_chunks(chunks: list[dict], label: str) -> str:
    """검색된 chunk 리스트를 LLM 컨텍스트용 문자열로 포맷팅."""
    if not chunks:
        return f"(검색된 {label} 근거 없음)"

    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        score = chunk.get("relevance_score", 0)
        cid = chunk.get("chroma_id", "N/A")
        content = (chunk.get("content") or "")[:500]
        source = meta.get("source_file") or meta.get("doc_filename") or "N/A"
        doc_type = meta.get("doc_type", "N/A")
        section = meta.get("article_title") or meta.get("section") or meta.get("section_title") or "N/A"
        lines.append(
            f"[{label} {i}] (유사도: {score}, ID: {cid})\n"
            f"  출처: {source} | 유형: {doc_type} | 섹션: {section}\n"
            f"  내용: {content}"
        )
    return "\n\n".join(lines)


def _query_to_str(q: str | list[str] | None, fallback: str) -> str:
    """search_queries 값(문자열 또는 리스트)을 단일 문자열로."""
    if q is None:
        return fallback
    if isinstance(q, str):
        return q.strip() or fallback
    if isinstance(q, list) and q:
        return (q[0].strip() if isinstance(q[0], str) else fallback) or fallback
    return fallback


# ───────────────────────────────────────────
# 노드 1: plan
# ───────────────────────────────────────────

def plan_node(state: ReviewState) -> dict:
    """Step 1: 위험 유형 분류 + 사용할 Tool + 검색 쿼리 생성."""
    start = time.time()

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        ("human", PLANNER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        plan = chain.invoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "broadcast_type": state.get("broadcast_type", "미지정"),
        })
        if not isinstance(plan, dict):
            plan = {}
    except Exception as e:
        logger.error("Plan 실패: %s", e)
        plan = {}

    if not plan:
        plan = {
            "risk_types": ["방송심의일반"],
            "tools_to_use": ["policy_search"],
            "search_queries": {"policy": state["item_text"], "cases": state["item_text"]},
        }

    elapsed = round(time.time() - start, 2)
    log = {"step": "plan", "risk_types": plan.get("risk_types", []), "elapsed": elapsed}

    return {
        "plan": plan,
        "retry_count": 0,
        "max_retries": 2,
        "tool_logs": [log],
    }


# ───────────────────────────────────────────
# 노드 2: retrieve
# ───────────────────────────────────────────

def retrieve_node(state: ReviewState) -> dict:
    """Step 2: Chroma DB에서 관련 문서 검색 (search_policy, search_cases)."""
    start = time.time()
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    tools_to_use = plan.get("tools_to_use", ["policy_search", "case_search"])
    item_text = state["item_text"]

    context: dict[str, list] = {
        "law_chunks": [],
        "regulation_chunks": [],
        "guideline_chunks": [],
        "case_chunks": [],
    }

    if "policy_search" in tools_to_use:
        policy_query = _query_to_str(queries.get("policy"), item_text)
        try:
            policy_result = search_policy.invoke({"query": policy_query})
            context["law_chunks"] = policy_result.get("law_chunks", [])
            context["regulation_chunks"] = policy_result.get("regulation_chunks", [])
            context["guideline_chunks"] = policy_result.get("guideline_chunks", [])
        except Exception as e:
            logger.error("policy 검색 실패: %s", e)

    if "case_search" in tools_to_use:
        case_query = _query_to_str(queries.get("cases"), item_text)
        try:
            case_result = search_cases.invoke({"query": case_query})
            context["case_chunks"] = case_result.get("case_chunks", [])
        except Exception as e:
            logger.error("case 검색 실패: %s", e)

    total_chunks = sum(len(v) for v in context.values())
    elapsed = round(time.time() - start, 2)
    log = {"step": "retrieve", "total_chunks": total_chunks, "elapsed": elapsed}

    return {"context": context, "tool_logs": [log]}


# ───────────────────────────────────────────
# 노드 3: grade_documents
# ───────────────────────────────────────────

def grade_documents_node(state: ReviewState) -> dict:
    """Step 3: 검색된 문서의 관련성을 LLM으로 배치 평가.

    chunk별 개별 LLM 호출(N회) → 전체를 한 번에 배치 평가(1회)로 변경.
    - 속도: N번 직렬 호출 제거, 단 1회 LLM 호출
    - CRAG의 "검색 결과 검증" 단계를 실질적으로 활성화
    """
    start = time.time()
    context = state.get("context", {})
    item_text = state["item_text"]
    risk_types = state.get("plan", {}).get("risk_types", [])
    risk_type = ", ".join(risk_types)

    # 1. 모든 chunk를 인덱스와 함께 수집
    all_chunks: list[tuple[str, int, dict]] = []
    for category_key in ["law_chunks", "regulation_chunks", "guideline_chunks", "case_chunks"]:
        for idx, chunk in enumerate(context.get(category_key, [])):
            all_chunks.append((category_key, idx, chunk))

    if not all_chunks:
        elapsed = round(time.time() - start, 2)
        return {
            "context": context,
            "relevant_doc_count": 0,
            "tool_logs": [{"step": "grade_documents", "total_evaluated": 0, "relevant_count": 0, "elapsed": elapsed}],
        }

    # 2. 배치 평가용 문서 목록 생성 (상위 20개 제한으로 토큰 초과 방지)
    chunks_to_grade = all_chunks[:10]
    documents_for_grading = []
    for i, (cat_key, idx, chunk) in enumerate(chunks_to_grade):
        content = (chunk.get("content") or "")[:500]
        source = chunk.get("metadata", {}).get("source_file", "N/A")
        doc_type = chunk.get("metadata", {}).get("doc_type", cat_key)
        documents_for_grading.append(
            f"[문서 {i + 1}] 유형: {doc_type} | 출처: {source}\n내용: {content}"
        )

    documents_text = "\n\n---\n\n".join(documents_for_grading)

    # 3. 배치 평가 프롬프트
    batch_grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 방송심의 문서 관련성 평가 전문가입니다.
        주어진 심의 대상 문구와 위험 유형에 대해, 각 문서가 심의 판단에 관련이 있는지 평가하세요.

        반드시 아래 JSON 형식으로만 응답하세요:
        {{
            "grades": [
                {{"doc_index": 1, "relevance": "relevant", "reason": "이유"}},
                {{"doc_index": 2, "relevance": "irrelevant", "reason": "이유"}}
            ]
        }}

        relevance는 반드시 "relevant" 또는 "irrelevant" 중 하나입니다."""),
                ("human", """## 심의 대상 문구
        {item_text}

        ## 위험 유형
        {risk_type}

        ## 평가 대상 문서들
        {documents_text}

        위 문서들 각각에 대해 심의 대상 문구와의 관련성을 평가해주세요."""),
    ])

    chain = batch_grade_prompt | _llm | _json_parser

    # 4. LLM 호출 (1회 배치)
    try:
        grade_result = chain.invoke({
            "item_text": item_text,
            "risk_type": risk_type,
            "documents_text": documents_text,
        })

        # 5. 관련 문서 인덱스 추출
        relevant_indices: set[int] = set()
        for g in grade_result.get("grades", []):
            if g.get("relevance") == "relevant":
                relevant_indices.add(g.get("doc_index", 0) - 1)  # 0-based

        filtered_context: dict[str, list] = {
            "law_chunks": [],
            "regulation_chunks": [],
            "guideline_chunks": [],
            "case_chunks": [],
        }
        relevant_count = 0
        for i, (cat_key, idx, chunk) in enumerate(chunks_to_grade):
            if i in relevant_indices:
                filtered_context[cat_key].append(chunk)
                relevant_count += 1

        # 상위 20개 제한으로 잘린 나머지는 원본 그대로 포함
        for cat_key, idx, chunk in all_chunks[20:]:
            filtered_context[cat_key].append(chunk)
            relevant_count += 1

        elapsed = round(time.time() - start, 2)
        log = {
            "step": "grade_documents",
            "method": "llm_batch",
            "total_evaluated": len(chunks_to_grade),
            "relevant_count": relevant_count,
            "filtered_out": len(chunks_to_grade) - len(relevant_indices),
            "elapsed": elapsed,
        }
        logger.info(
            "grade_documents: %d개 중 %d개 관련 문서 선별 (%.2fs)",
            len(chunks_to_grade), relevant_count, elapsed,
        )

        return {
            "context": filtered_context,
            "relevant_doc_count": relevant_count,
            "tool_logs": [log],
        }

    except Exception as e:
        # 에러 시 보수적으로 전부 포함
        logger.error("grade_documents LLM 호출 실패: %s", e)
        elapsed = round(time.time() - start, 2)
        total = sum(len(v) for v in context.values() if isinstance(v, list))
        return {
            "context": context,
            "relevant_doc_count": total,
            "tool_logs": [{
                "step": "grade_documents",
                "method": "fallback_passthrough",
                "error": str(e),
                "total_evaluated": total,
                "relevant_count": total,
                "elapsed": elapsed,
            }],
        }


# ───────────────────────────────────────────
# 노드 4: rewrite_query
# ───────────────────────────────────────────

def rewrite_query_node(state: ReviewState) -> dict:
    """Step 4: 검색 쿼리 재작성 후 재검색용 plan 갱신."""
    start = time.time()
    plan = state.get("plan", {})
    queries = plan.get("search_queries", {})
    item_text = state["item_text"]

    old_policy = _query_to_str(queries.get("policy"), item_text)
    old_case = _query_to_str(queries.get("cases"), item_text)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_QUERY_SYSTEM),
        ("human", REWRITE_QUERY_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        new_queries = chain.invoke({
            "item_text": item_text,
            "risk_type": ", ".join(plan.get("risk_types", [])),
            "old_policy_query": old_policy,
            "old_case_query": old_case,
        })
        if not isinstance(new_queries, dict):
            new_queries = {}
    except Exception as e:
        logger.error("쿼리 재작성 실패: %s", e)
        new_queries = {}

    policy_query = new_queries.get("policy_query", item_text)
    case_query = new_queries.get("case_query", item_text)
    updated_plan = dict(plan)
    updated_plan["search_queries"] = {"policy": policy_query, "cases": case_query}

    retry_count = state.get("retry_count", 0) + 1
    elapsed = round(time.time() - start, 2)
    log = {
        "step": "rewrite_query",
        "retry_count": retry_count,
        "new_policy_query": policy_query,
        "new_case_query": case_query,
        "elapsed": elapsed,
    }

    return {"plan": updated_plan, "retry_count": retry_count, "tool_logs": [log]}


# ───────────────────────────────────────────
# 노드 5: generate
# ───────────────────────────────────────────

def generate_node(state: ReviewState) -> dict:
    """Step 5: 검색된 근거를 바탕으로 최종 심의 의견 생성."""
    start = time.time()
    plan = state.get("plan", {})
    context = state.get("context", {})

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM),
        ("human", GENERATOR_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    risk_types = plan.get("risk_types", ["방송심의일반"])
    default_result = {
        "judgment": "주의",
        "reason": "",
        "risk_type": risk_types[0] if risk_types else "",
        "related_articles": [],
        "suggested_fix": "",
        "references": [],
    }

    try:
        result = chain.invoke({
            "item_text": state["item_text"],
            "category": state.get("category", "미지정"),
            "risk_type": ", ".join(risk_types),
            "law_context": _format_chunks(context.get("law_chunks", []), "법률"),
            "regulation_context": _format_chunks(context.get("regulation_chunks", []), "규정"),
            "guideline_context": _format_chunks(context.get("guideline_chunks", []), "지침"),
            "case_context": _format_chunks(context.get("case_chunks", []), "사례"),
        })
        if isinstance(result, dict):
            default_result.update(result)
            result = default_result
        else:
            result = default_result
    except Exception as e:
        logger.error("Generate 실패: %s", e)
        default_result["reason"] = f"AI 생성 중 오류 발생: {str(e)}"
        result = default_result

    elapsed = round(time.time() - start, 2)
    log = {"step": "generate", "judgment": result.get("judgment", ""), "elapsed": elapsed}

    return {"result": result, "tool_logs": [log]}


# ───────────────────────────────────────────
# 노드 6: grade_answer
# ───────────────────────────────────────────

def grade_answer_node(state: ReviewState) -> dict:
    """Step 6: 생성된 심의 의견의 품질을 LLM으로 평가."""
    start = time.time()
    result = state.get("result", {})
    context = state.get("context", {})

    context_summary_parts: list[str] = []
    for key in ("law_chunks", "regulation_chunks", "guideline_chunks", "case_chunks"):
        for chunk in context.get(key, []):
            context_summary_parts.append((chunk.get("content") or "")[:200])
    context_summary_text = "\n---\n".join(context_summary_parts) if context_summary_parts else "(검색된 근거 없음)"
    context_summary_text = context_summary_text[:2000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADE_ANSWER_SYSTEM),
        ("human", GRADE_ANSWER_HUMAN),
    ])
    chain = prompt | _llm | _json_parser

    try:
        grade = chain.invoke({
            "item_text": state["item_text"],
            "answer_json": json.dumps(result, ensure_ascii=False, indent=2),
            "context_summary": context_summary_text,
        })
        answer_grade = grade.get("grade", "pass") if isinstance(grade, dict) else "pass"
    except Exception as e:
        logger.warning("답변 평가 실패, pass 처리: %s", e)
        answer_grade = "pass"

    elapsed = round(time.time() - start, 2)
    log = {"step": "grade_answer", "grade": answer_grade, "elapsed": elapsed}

    return {"answer_grade": answer_grade, "tool_logs": [log]}


# ───────────────────────────────────────────
# 라우터
# ───────────────────────────────────────────

def route_after_grade_documents(state: ReviewState) -> str:
    """grade_documents 결과에 따라 다음 노드 결정."""
    relevant_count = state.get("relevant_doc_count", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if relevant_count >= 1:
        return "generate"
    if retry_count >= max_retries:
        logger.warning("재시도 %s회 초과. 현재 문서로 generate 진행.", max_retries)
        return "generate"
    return "rewrite_query"


def route_after_grade_answer(state: ReviewState) -> str:
    """grade_answer 결과에 따라 다음 노드 결정."""
    answer_grade = state.get("answer_grade", "pass")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if answer_grade == "pass":
        return "end"
    if retry_count >= max_retries:
        logger.warning("재시도 %s회 초과. 현재 답변으로 종료.", max_retries)
        return "end"
    return "rewrite_query"


# ───────────────────────────────────────────
# 그래프 구성
# ───────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(ReviewState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_answer", grade_answer_node)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "grade_answer")

    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {"end": END, "rewrite_query": "rewrite_query"},
    )

    return workflow.compile()


graph = _build_graph()


# ───────────────────────────────────────────
# ReviewChain (인터페이스 유지)
# ───────────────────────────────────────────

class ReviewChain:
    """
    방송 심의 Self-Corrective RAG 워크플로우.

    LangGraph: plan → retrieve → grade_documents → (rewrite_query 루프) → generate → grade_answer → (루프) → end

    rag_service.py 호출: ReviewChain(model_name).run(item_text, category, broadcast_type) → dict
    """

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.model_name = model_name

    def run(
        self,
        item_text: str,
        category: str = "",
        broadcast_type: str = "",
    ) -> dict:
        """기존과 동일한 인터페이스로 LangGraph 그래프를 실행."""
        initial_state: ReviewState = {
            "item_text": item_text,
            "category": category or "미지정",
            "broadcast_type": broadcast_type or "미지정",
            "plan": {},
            "context": {},
            "result": {},
            "relevant_doc_count": 0,
            "answer_grade": "",
            "retry_count": 0,
            "max_retries": 3,
            "tool_logs": [],
        }

        try:
            run_label = item_text[:40].replace("\n", " ")
            final_state = graph.invoke(
                initial_state,
                config=RunnableConfig(
                    run_name=f"review_chain:{run_label}",
                    tags=["review-chain", "langgraph", broadcast_type or "미지정"],
                    metadata={
                        "item_text": item_text,
                        "category": category,
                        "broadcast_type": broadcast_type,
                    },
                ),
            )
            output = dict(final_state.get("result", {}))
            output["tool_logs"] = final_state.get("tool_logs", [])
            return output
        except Exception as e:
            logger.error("Graph 실행 실패: %s", e)
            return {
                "judgment": "주의",
                "reason": f"AI 처리 중 오류 발생: {str(e)}",
                "risk_type": "",
                "related_articles": [],
                "suggested_fix": "",
                "references": [],
                "tool_logs": [{"step": "error", "message": str(e)}],
            }
