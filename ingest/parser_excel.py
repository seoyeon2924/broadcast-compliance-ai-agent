"""
Excel parser using openpyxl.
컬럼 A/B/C를 별도 필드로 반환 (섹션 분리·메타데이터는 chunker에서 처리).
"""

from pathlib import Path

from openpyxl import load_workbook


class ExcelParser:
    """Parse an Excel (.xlsx) file; returns per-row dict with separate columns."""

    @staticmethod
    def parse(file_path: str) -> list[dict]:
        """
        Returns:
            List of {
                "opinion_note": str,      # 컬럼A 원문
                "violation_type": str,    # 컬럼B
                "limit_expression": str,  # 컬럼C
                "row": int,
                "sheet": str,
                "source_file": str,
            }
        """
        path = Path(file_path)
        wb = load_workbook(str(path), read_only=True, data_only=True)

        results: list[dict] = []
        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                for row_idx, row in enumerate(
                    ws.iter_rows(min_row=2, values_only=True), start=2
                ):
                    col_a = str(row[0] or "").strip() if len(row) > 0 else ""
                    col_b = str(row[1] or "").strip() if len(row) > 1 else ""
                    col_c = str(row[2] or "").strip() if len(row) > 2 else ""

                    if not col_a and not col_b:
                        continue

                    results.append(
                        {
                            "opinion_note": col_a,
                            "violation_type": col_b,
                            "limit_expression": col_c,
                            "row": row_idx,
                            "sheet": sheet_name,
                            "source_file": path.name,
                        }
                    )
        finally:
            wb.close()
        return results
