"""
Excel parser using openpyxl.
Stage 2 implementation.
"""


class ExcelParser:
    @staticmethod
    def parse(file_path: str) -> list[dict]:
        """
        Extract rows from an Excel file with sheet/row metadata.

        Returns:
            list of {"sheet": str, "row": int, "text": str}
        """
        # TODO: Stage 2 — implement with openpyxl
        raise NotImplementedError(
            "Excel parsing will be implemented in Stage 2"
        )
