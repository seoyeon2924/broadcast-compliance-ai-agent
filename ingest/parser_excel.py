"""
Excel parser using openpyxl.
컬럼 A~E를 별도 필드로 반환 (섹션 분리·메타데이터는 chunker에서 처리).

컬럼 구조:
  A: 처리번호 (case_number)
  B: 처리일자 (case_date)
  C: 심의의견 (opinion_note)
  D: 심의지적코드 (violation_type)
  E: 한정표현 (limit_expression)
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
                "case_number": str,       # 컬럼A 처리번호
                "case_date": str,         # 컬럼B 처리일자
                "opinion_note": str,      # 컬럼C 심의의견 원문
                "violation_type": str,    # 컬럼D 심의지적코드
                "limit_expression": str,  # 컬럼E 한정표현
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
                    col_d = str(row[3] or "").strip() if len(row) > 3 else ""
                    col_e = str(row[4] or "").strip() if len(row) > 4 else ""

                    # 처리번호·심의의견 중 하나라도 있으면 포함
                    if not col_a and not col_c:
                        continue

                    # 처리일자: datetime 객체로 읽히는 경우 날짜 부분만 추출
                    case_date = col_b
                    if " " in col_b:
                        case_date = col_b.split(" ")[0]

                    # 처리번호: 소수점 제거 (openpyxl이 숫자로 읽는 경우)
                    case_number = col_a.split(".")[0] if "." in col_a else col_a

                    results.append(
                        {
                            "case_number": case_number,
                            "case_date": case_date,
                            "opinion_note": col_c,
                            "violation_type": col_d,
                            "limit_expression": col_e,
                            "row": row_idx,
                            "sheet": sheet_name,
                            "source_file": path.name,
                        }
                    )
        finally:
            wb.close()
        return results
