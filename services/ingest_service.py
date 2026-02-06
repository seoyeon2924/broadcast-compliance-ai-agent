"""
Ingest Service — file upload, parsing, chunking, indexing.

Stage 1: saves the file and creates dummy chunks.
Stage 2: real parsing (PDF/Excel/DOCX) + Chroma embedding.
"""

from datetime import datetime

from config import settings
from storage.models import DocStatus
from storage.repository import DocumentRepository, AuditRepository


class IngestService:

    @staticmethod
    def upload_and_index(
        file,
        doc_type: str,
        category: str,
        scope: str,
        uploaded_by: str,
    ) -> dict:
        """
        Stage 1 implementation:
          1. Save uploaded file to data/uploads/
          2. Create ReferenceDocument (UPLOADED → INDEXING → INDEXED)
          3. Create dummy chunks (real parsing in Stage 2)
        """
        # ── Save file ──
        file_path = settings.UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # ── Create document record ──
        doc_id = DocumentRepository.create(
            filename=file.name,
            doc_type=doc_type,
            category=category,
            scope=scope,
            file_path=str(file_path),
            uploaded_by=uploaded_by,
        )

        # ── Transition: UPLOADED → INDEXING ──
        DocumentRepository.update_status(doc_id, DocStatus.INDEXING.value)

        try:
            # ── Stage 1: dummy chunks ──
            dummy_chunks = []
            dummy_count = 3
            for i in range(dummy_count):
                dummy_chunks.append(
                    {
                        "chunk_index": i,
                        "content_preview": (
                            f"[단계1 더미 청크 {i + 1}] "
                            f"{file.name} 내용 미리보기..."
                        ),
                        "page_or_row": f"p.{i + 1}",
                        "source_file": file.name,
                        "doc_type": doc_type,
                        "chroma_id": f"{doc_id}_chunk_{i}",
                    }
                )
            chunk_count = DocumentRepository.create_chunks(doc_id, dummy_chunks)

            # ── Transition: INDEXING → INDEXED ──
            DocumentRepository.update_status(
                doc_id,
                DocStatus.INDEXED.value,
                chunk_count=chunk_count,
                indexed_at=datetime.utcnow(),
            )

            AuditRepository.create_log(
                event_type="INGEST",
                entity_type="ReferenceDocument",
                entity_id=doc_id,
                actor=uploaded_by,
                detail={
                    "filename": file.name,
                    "chunk_count": chunk_count,
                    "stage": "stage1-dummy",
                },
            )

            return {
                "doc_id": doc_id,
                "filename": file.name,
                "chunk_count": chunk_count,
                "status": DocStatus.INDEXED.value,
            }

        except Exception as e:
            DocumentRepository.update_status(
                doc_id,
                DocStatus.INDEX_FAILED.value,
                error_message=str(e),
            )
            raise

    @staticmethod
    def list_documents() -> list[dict]:
        return DocumentRepository.list_all()

    @staticmethod
    def generate_advanced_metadata(document_id: str) -> dict:
        """
        Stage 2: LLM-based section_title / keywords extraction.
        Stage 1: placeholder.
        """
        # TODO: Stage 2 — iterate over chunks, call LLM, update Chroma meta
        return {
            "document_id": document_id,
            "status": "NOT_IMPLEMENTED",
            "message": "고급 메타데이터 생성은 단계2에서 구현됩니다.",
        }
