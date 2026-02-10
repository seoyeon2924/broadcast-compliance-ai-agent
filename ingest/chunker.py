"""
Structure-aware chunker — 문서 유형별 조/항/섹션 인식 + RecursiveCharacterTextSplitter.

유형: law, regulation (조 단위), guideline (대분류+소항목), case (Excel 섹션 분리).
"""

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


# ── 법률/규정: 조(條) 단위 분리 ──
ARTICLE_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?\s*(?:\([^)]*\))?",
    re.UNICODE,
)
CHAPTER_PATTERN = re.compile(
    r"제(\d+)장\s*(.+?)(?=\n|$)",
    re.UNICODE,
)
SECTION_PATTERN = re.compile(
    r"제(\d+)절\s*(.+?)(?=\n|$)",
    re.UNICODE,
)
ARTICLE_TITLE_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?\s*\(([^)]+)\)",
    re.UNICODE,
)

# ── 지침: 대분류(Ⅰ~Ⅹ) ──
MAJOR_PATTERN = re.compile(
    r"(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ|Ⅷ|Ⅸ|Ⅹ)\.\s*(.+?)(?=\n|$)",
    re.UNICODE,
)

# ── 사례(Excel): 컬럼A 섹션 구분 ──
CASE_SECTION_PATTERNS = [
    re.compile(r"[●■]\s*상품\s*정보", re.IGNORECASE),
    re.compile(r"[●■]\s*심의\s*의견", re.IGNORECASE),
    re.compile(r"[●■]\s*수정\s*사항", re.IGNORECASE),
    re.compile(r"[●■]\s*주의\s*사항", re.IGNORECASE),
]


def _truncate(s: str, max_len: int = 500) -> str:
    """Chroma 메타 길이 제한."""
    if not s:
        return ""
    s = str(s).strip()
    return s[:max_len] if len(s) > max_len else s


class Chunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def _secondary_split(
        self,
        text: str,
        base_meta: dict[str, Any],
        source_file: str,
        base_index: int,
    ) -> list[dict]:
        """긴 텍스트를 RecursiveCharacterTextSplitter로 2차 분할."""
        if len(text) <= self.chunk_size:
            return [
                {
                    **base_meta,
                    "content": text,
                    "source_file": source_file,
                    "chunk_index": base_index,
                }
            ]
        parts = self._splitter.split_text(text)
        out = []
        for i, part in enumerate(parts):
            page_or_row = base_meta.get("page_or_row", "")
            if len(parts) > 1:
                page_or_row = f"{page_or_row}_part{i + 1}"
            out.append({
                **base_meta,
                "page_or_row": page_or_row,
                "content": part,
                "source_file": source_file,
                "chunk_index": base_index + i,
            })
        return out

    # ── 법률 / 규정: 조 단위 ──

    def chunk_law(self, full_text: str, source_file: str) -> list[dict]:
        """법률: 조 단위 1차 분리 → 2차 RecursiveCharacterTextSplitter."""
        return self._chunk_by_article(full_text, source_file, "law")

    def chunk_regulation(self, full_text: str, source_file: str) -> list[dict]:
        """규정: chunk_law와 동일, doc_structure_type만 regulation."""
        return self._chunk_by_article(full_text, source_file, "regulation")

    def _chunk_by_article(
        self,
        full_text: str,
        source_file: str,
        doc_structure_type: str,
    ) -> list[dict]:
        parts: list[tuple[int, int, str, str, str]] = []  # start, end, label, num, title
        for m in ARTICLE_PATTERN.finditer(full_text):
            label = m.group(0).strip()
            article_num = re.match(r"(제\d+조(?:의\d+)?)", label)
            article_number = article_num.group(1) if article_num else label
            article_title = ""
            tit = ARTICLE_TITLE_PATTERN.search(label)
            if tit:
                article_title = tit.group(1).strip()
            parts.append((m.start(), m.end(), label, article_number, article_title))

        if not parts:
            return self.chunk_fallback(full_text, source_file)

        chunks_out: list[dict] = []
        chunk_index = 0
        for i, (start, end, label, article_number, article_title) in enumerate(parts):
            next_start = parts[i + 1][0] if i + 1 < len(parts) else len(full_text)
            block = full_text[start:next_start].strip()
            if not block:
                continue

            preceding = full_text[:start]
            chapter = ""
            section = ""
            for cm in CHAPTER_PATTERN.finditer(preceding):
                chapter = f"제{cm.group(1)}장 {cm.group(2).strip()}"
            for sm in SECTION_PATTERN.finditer(preceding):
                section = f"제{sm.group(1)}절 {sm.group(2).strip()}"

            article_prefix = re.match(r"제\d+조(?:의\d+)?", label)
            page_or_row = article_prefix.group(0) if article_prefix else label
            base_meta = {
                "page_or_row": page_or_row,
                "doc_structure_type": doc_structure_type,
                "chapter": _truncate(chapter),
                "section": _truncate(section),
                "article_number": article_number,
                "article_title": _truncate(article_title),
                "major_section": "",
                "sub_section": "",
                "sub_detail": "",
                "violation_type": "",
                "limit_expression": "",
                "product_summary": "",
            }

            sub_chunks = self._secondary_split(
                block, base_meta, source_file, chunk_index
            )
            for sc in sub_chunks:
                sc["chunk_index"] = chunk_index
                chunk_index += 1
            chunks_out.extend(sub_chunks)

        for i, c in enumerate(chunks_out):
            c["chunk_index"] = i
        return chunks_out

    # ── 지침: 대분류 + 소항목 ──

    def chunk_guideline(self, full_text: str, source_file: str) -> list[dict]:
        """지침: 대분류(Ⅰ~Ⅹ) 내 소항목(1., 2., 3.) 단위 분리."""
        # 1차: 대분류 + 소항목 경계로 블록 나누기 (간단히 \n\n 또는 Ⅴ. Ⅵ. 등으로)
        blocks: list[dict] = []  # {major, sub_section, sub_detail, text}
        lines = full_text.split("\n")
        current_major = ""
        current_sub = ""
        current_detail = ""
        current_text: list[str] = []
        sub_pattern = re.compile(r"^\s*(\d+)\.\s+")
        detail_pattern = re.compile(r"^\s*(가|나|다|라|마|바|사)\.\s+")

        def flush():
            if current_text:
                text = "\n".join(current_text).strip()
                if text:
                    blocks.append({
                        "major_section": current_major,
                        "sub_section": current_sub,
                        "sub_detail": current_detail,
                        "text": text,
                    })

        for line in lines:
            major_m = MAJOR_PATTERN.match(line.strip())
            if major_m:
                flush()
                current_major = f"{major_m.group(1)}. {major_m.group(2).strip()}"
                current_sub = ""
                current_detail = ""
                current_text = [line]
                continue
            sub_m = sub_pattern.match(line)
            if sub_m and current_major:
                flush()
                current_sub = line.strip()
                current_detail = ""
                current_text = [line]
                continue
            detail_m = detail_pattern.match(line)
            if detail_m and current_sub:
                flush()
                current_detail = line.strip()
                current_text = [line]
                continue
            if current_major or current_text:
                current_text.append(line)

        flush()

        if not blocks:
            return self.chunk_fallback(full_text, source_file)

        chunks_out = []
        for idx, blk in enumerate(blocks):
            base_meta = {
                "page_or_row": f"{blk['major_section'][:2]}-{idx + 1}" if blk["major_section"] else str(idx + 1),
                "doc_structure_type": "guideline",
                "chapter": "",
                "section": "",
                "article_number": "",
                "article_title": "",
                "major_section": _truncate(blk["major_section"]),
                "sub_section": _truncate(blk["sub_section"]),
                "sub_detail": _truncate(blk["sub_detail"]),
                "violation_type": "",
                "limit_expression": "",
                "product_summary": "",
            }
            sub_chunks = self._secondary_split(
                blk["text"], base_meta, source_file, idx
            )
            for i, sc in enumerate(sub_chunks):
                sc["chunk_index"] = len(chunks_out) + i
                if len(sub_chunks) > 1:
                    sc["page_or_row"] = f"{base_meta['page_or_row']}_part{i + 1}"
            chunks_out.extend(sub_chunks)

        for i, c in enumerate(chunks_out):
            c["chunk_index"] = i
        return chunks_out

    # ── 과거 심의 사례 (Excel) ──

    def chunk_cases(self, rows: list[dict]) -> list[dict]:
        """과거 심의 지적 사례: 컬럼A 섹션 분리, 심의의견+수정+주의 → content, 상품정보 앞 100자 → product_summary."""
        chunks_out: list[dict] = []
        for row_idx, row in enumerate(rows):
            opinion = row.get("opinion_note", "") or ""
            violation_type = row.get("violation_type", "") or ""
            limit_expr = row.get("limit_expression", "") or ""
            sheet = row.get("sheet", "")
            row_num = row.get("row", row_idx + 2)
            source_file = row.get("source_file", "")

            product_summary = ""
            content_parts: list[str] = []
            positions: list[tuple[int, int, str]] = []
            for pat in CASE_SECTION_PATTERNS:
                for m in pat.finditer(opinion):
                    positions.append((m.start(), m.end(), m.group(0).strip()))
            positions.sort(key=lambda x: x[0])

            if not positions:
                content = opinion
            else:
                for i, (start, end, label) in enumerate(positions):
                    seg_end = positions[i + 1][0] if i + 1 < len(positions) else len(opinion)
                    segment = opinion[end:seg_end].strip()
                    if "상품" in label and "정보" in label:
                        product_summary = segment[:100]
                    else:
                        if segment:
                            content_parts.append(segment)
                content = "\n\n".join(content_parts) if content_parts else opinion

            if not content.strip():
                content = opinion

            base_meta = {
                "page_or_row": f"{sheet}:Row{row_num}",
                "source_file": source_file,
                "doc_structure_type": "case",
                "chapter": "",
                "section": "",
                "article_number": "",
                "article_title": "",
                "major_section": "",
                "sub_section": "",
                "sub_detail": "",
                "violation_type": _truncate(violation_type),
                "limit_expression": _truncate(limit_expr),
                "product_summary": _truncate(product_summary, 100),
            }
            sub_chunks = self._secondary_split(
                content, base_meta, source_file, len(chunks_out)
            )
            for i, sc in enumerate(sub_chunks):
                sc["chunk_index"] = len(chunks_out) + i
                sc["page_or_row"] = base_meta["page_or_row"]
                if len(sub_chunks) > 1:
                    sc["page_or_row"] = f"{base_meta['page_or_row']}_part{i + 1}"
            chunks_out.extend(sub_chunks)

        for i, c in enumerate(chunks_out):
            c["chunk_index"] = i
        return chunks_out

    # ── Fallback ──

    def chunk_fallback(self, full_text: str, source_file: str) -> list[dict]:
        """패턴 매칭 실패 시 RecursiveCharacterTextSplitter만 사용."""
        parts = self._splitter.split_text(full_text)
        return [
            {
                "content": p,
                "page_or_row": f"p.{i + 1}",
                "source_file": source_file,
                "chunk_index": i,
                "doc_structure_type": "",
                "chapter": "",
                "section": "",
                "article_number": "",
                "article_title": "",
                "major_section": "",
                "sub_section": "",
                "sub_detail": "",
                "violation_type": "",
                "limit_expression": "",
                "product_summary": "",
            }
            for i, p in enumerate(parts)
        ]

    # ── 하위 호환 ──

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        """기존 PDF 청킹 — 페이지 단위 고정 크기. 하위 호환용."""
        full_text = "\n".join(p["text"] for p in pages)
        return self.chunk_fallback(full_text, pages[0]["source_file"] if pages else "")

    def chunk_rows(self, rows: list[dict]) -> list[dict]:
        """기존 Excel 청킹 — chunk_cases로 위임. opinion_note 등 없으면 기존 형식 가정."""
        if rows and "opinion_note" in rows[0]:
            return self.chunk_cases(rows)
        # 레거시: text 필드만 있는 경우
        return [
            {
                "content": r.get("text", ""),
                "page_or_row": f"{r.get('sheet', '')}:Row{r.get('row', i)}",
                "source_file": r.get("source_file", ""),
                "chunk_index": i,
                "doc_structure_type": "case",
                "chapter": "",
                "section": "",
                "article_number": "",
                "article_title": "",
                "major_section": "",
                "sub_section": "",
                "sub_detail": "",
                "violation_type": "",
                "limit_expression": "",
                "product_summary": "",
            }
            for i, r in enumerate(rows)
        ]
