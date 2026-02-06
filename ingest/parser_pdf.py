"""
PDF parser using PyMuPDF (fitz).
Stage 2 implementation.
"""


class PDFParser:
    @staticmethod
    def parse(file_path: str) -> list[dict]:
        """
        Extract text from a PDF with page metadata.

        Returns:
            list of {"page": int, "text": str}
        """
        # TODO: Stage 2 — implement with PyMuPDF (fitz)
        raise NotImplementedError("PDF parsing will be implemented in Stage 2")
