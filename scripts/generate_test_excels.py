from pathlib import Path
import pandas as pd

# Determine repo root (scripts/ is one level under repo root)
repo_root = Path(__file__).resolve().parents[1]
input_dir = repo_root / "input"
input_dir.mkdir(parents=True, exist_ok=True)

# 1) Simple table-like workbook
table_like_20_rows = pd.DataFrame({
    "Name": ["Alice","Bob","Charlie","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Karl","Liam","Mia","Nina","Oscar","Peggy","Quinn","Rita","Sam","Tina"],
    "Hours": [8, 7.5, 6,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "Project": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B"]
})

# 2) Mixed-content workbook: one sheet with clear header, one sheet without headers
table_like = pd.DataFrame({
    "Name": ["Alice","Bob","Charlie"],
    "Hours": [8, 7.5, 6],
    "Project": ["A", "B", "A"]
})

no_header = pd.DataFrame([
    ["Total", "", ""],
    ["2025-09-28", "Invoice", 1234],
    ["", "", ""],
    ["Data", 1, 2],
])

with pd.ExcelWriter(input_dir / "Mixed_test.xlsx", engine="openpyxl") as ew:
    table_like_20_rows.to_excel(ew, index=False, sheet_name="TableLike20Rows")
    table_like.to_excel(ew, index=False, sheet_name="TableLike")
    no_header.to_excel(ew, index=False, header=False, sheet_name="NoHeaders")

print(f"Generated test excels in: {input_dir}")