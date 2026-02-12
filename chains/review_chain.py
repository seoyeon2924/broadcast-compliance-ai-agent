"""
ReviewChain — 3단계 파이프라인 (Plan → Retrieve → Generate).

Phase 1 Tool + Phase 2 Prompt 조합.
"""

from __future__ import annotations

import time

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import settings
from prompts.planner import PLANNER_SYSTEM, PLANNER_HUMAN
from prompts.generator import GENERATOR_SYSTEM, GENERATOR_HUMAN
from tools.policy_tools import search_policy
from tools.case_tools import search_cases


def _dedup_chunks_by_id(chunks: list[dict]) -> list[dict]:
    """chroma_id 기준으로 중복 제거 (첫 번째 유지)."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        cid = c.get("chroma_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def _normalize_queries(queries: str | list[str] | None) -> list[str]:
    """search_queries value를 리스트로 통일."""
    if queries is None:
        return []
    if isinstance(queries, str):
        return [queries] if queries.strip() else []
    return [q for q in queries if isinstance(q, str) and q.strip()]


class ReviewChain:
    """Plan → Retrieve → Generate 3단계 고정 워크플로우."""

    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY or "dummy",
            temperature=0,
            request_timeout=90,
            http_client=httpx.Client(verify=False),
        )
        self.json_parser = JsonOutputParser()
        self.model_name = model_name

    def run(
        self,
        item_text: str,
        category: str = "",
        broadcast_type: str = "",
    ) -> dict:
        """메인 진입점. 3단계 순차 실행 후 결과 합침."""
        tool_logs: list[dict] = []

        # Step 1: Plan
        plan = self._step_plan(
            item_text,
            category or "미지정",
            broadcast_type or "미지정",
        )
        tool_logs.append({"step": "plan", "result": plan})

        # Step 2: Retrieve
        retrieve_result = self._step_retrieve(plan)
        context = retrieve_result["context"]
        tool_logs.append({
            "step": "retrieve",
            "tools_called": retrieve_result["tools_called"],
            "chunks_found": retrieve_result["chunks_found"],
        })

        # Step 3: Generate
        gen = self._step_generate(
            item_text,
            category or "미지정",
            plan,
            context,
        )

        return {
            "judgment": gen.get("judgment", "주의"),
            "reason": gen.get("reason", ""),
            "risk_type": gen.get("risk_type", ""),
            "related_articles": gen.get("related_articles", []),
            "suggested_fix": gen.get("suggested_fix", ""),
            "references": gen.get("references", []),
            "tool_logs": tool_logs,
        }

    def _step_plan(
        self,
        item_text: str,
        category: str,
        broadcast_type: str,
    ) -> dict:
        """Planner: 위험 유형 분류 및 검색 쿼리 결정."""
        default_plan = {
            "risk_types": ["방송심의일반"],
            "risk_keywords": [],
            "risk_analysis": "",
            "tools_to_use": ["policy_search", "case_search"],
            "search_queries": {
                "policy": [item_text],
                "cases": [item_text],
            },
        }
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", PLANNER_SYSTEM),
                ("human", PLANNER_HUMAN),
            ])
            chain = prompt | self.llm | self.json_parser
            result = chain.invoke({
                "item_text": item_text,
                "category": category,
                "broadcast_type": broadcast_type,
            })
            if isinstance(result, dict):
                return result
            return default_plan
        except Exception:
            return default_plan

    def _step_retrieve(self, plan: dict) -> dict:
        """Retrieve: plan에 따라 policy_search / case_search 호출, context 구성."""
        context = {
            "law_chunks": [],
            "regulation_chunks": [],
            "guideline_chunks": [],
            "case_chunks": [],
        }
        tools_called: list[str] = []
        chunks_found: dict = {}

        tools_to_use = plan.get("tools_to_use") or ["policy_search"]
        search_queries = plan.get("search_queries") or {}

        # Policy search
        if "policy_search" in tools_to_use:
            queries = _normalize_queries(search_queries.get("policy"))
            if not queries:
                queries = [plan.get("risk_analysis", "") or "방송 광고 심의 기준"]
            for q in queries:
                try:
                    out = search_policy.invoke({"query": q})
                    for key in ("law_chunks", "regulation_chunks", "guideline_chunks"):
                        context[key] = _dedup_chunks_by_id(
                            context[key] + out.get(key, [])
                        )
                except Exception:
                    pass
            tools_called.append("policy_search")
            chunks_found["law"] = len(context["law_chunks"])
            chunks_found["regulation"] = len(context["regulation_chunks"])
            chunks_found["guideline"] = len(context["guideline_chunks"])

        # Case search
        if "case_search" in tools_to_use:
            queries = _normalize_queries(search_queries.get("cases"))
            if not queries:
                queries = [plan.get("risk_analysis", "") or "방송 광고 심의 사례"]
            for q in queries:
                try:
                    out = search_cases.invoke({"query": q})
                    context["case_chunks"] = _dedup_chunks_by_id(
                        context["case_chunks"] + out.get("case_chunks", [])
                    )
                except Exception:
                    pass
            tools_called.append("case_search")
            chunks_found["cases"] = len(context["case_chunks"])

        return {
            "context": context,
            "tools_called": tools_called,
            "chunks_found": chunks_found,
        }

    def _step_generate(
        self,
        item_text: str,
        category: str,
        plan: dict,
        context: dict,
    ) -> dict:
        """Generator: 검색된 근거로 최종 심의 의견 생성."""
        risk_types = plan.get("risk_types") or ["방송심의일반"]
        default_gen = {
            "judgment": "주의",
            "reason": "",
            "risk_type": risk_types[0] if risk_types else "",
            "related_articles": [],
            "suggested_fix": "",
            "references": [],
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM),
            ("human", GENERATOR_HUMAN),
        ])
        chain = prompt | self.llm | self.json_parser
        invoke_kw = {
            "item_text": item_text,
            "category": category,
            "risk_type": ", ".join(risk_types),
            "law_context": self._format_chunks(context.get("law_chunks", []), "법률"),
            "regulation_context": self._format_chunks(
                context.get("regulation_chunks", []), "규정"
            ),
            "guideline_context": self._format_chunks(
                context.get("guideline_chunks", []), "지침"
            ),
            "case_context": self._format_chunks(
                context.get("case_chunks", []), "사례"
            ),
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                result = chain.invoke(invoke_kw)
                if isinstance(result, dict):
                    return result
                return default_gen
            except Exception as e:
                last_error = e
                if attempt < 2 and (
                    "Connection" in str(e)
                    or "timeout" in str(e).lower()
                    or "ConnectionError" in type(e).__name__
                ):
                    time.sleep(2)
                    continue
                break
        default_gen["reason"] = f"AI 생성 중 오류 발생: {str(last_error)}"
        return default_gen

    @staticmethod
    def _format_chunks(chunks: list[dict], label: str) -> str:
        """검색된 chunk 리스트를 LLM 컨텍스트용 문자열로 포맷팅."""
        if not chunks:
            return f"(검색된 {label} 근거 없음)"
        lines: list[str] = []
        for i, c in enumerate(chunks, 1):
            cid = c.get("chroma_id", "")
            score = c.get("relevance_score", 0)
            meta = c.get("metadata") or {}
            source = meta.get("source_file") or meta.get("doc_filename") or ""
            doc_type = meta.get("doc_type") or ""
            section = (
                meta.get("article_title")
                or meta.get("section")
                or meta.get("section_title")
                or ""
            )
            content = (c.get("content") or "")[:500]
            lines.append(
                f"[{label} {i}] (유사도: {score}, ID: {cid})\n"
                f"  출처: {source} | 유형: {doc_type} | 섹션: {section}\n"
                f"  내용: {content}"
            )
        return "\n\n".join(lines)
