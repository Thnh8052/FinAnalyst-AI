"""
linearizer.py

TableLinearizer:
- Chuyển đổi ma trận bảng tài chính 2D thành chuỗi quan hệ bộ ba ngữ nghĩa (Semantic Tuples):
    [Chỉ tiêu, Kỳ/Năm, Giá trị, Đơn vị]
- Hỗ trợ phân cấp hàng Cha - Con (Parent-Child Row Indentation):
    - Nhận diện hàng Header nhóm (hàng chỉ có text ở Cột 0, toàn bộ cột số trống).
    - Làm phẳng hàng con thành: "Operating expenses > Research and development | FY2025: $12,914 million".
    - Nhận diện hàng Total/Subtotal để reset cấp bậc.
- Phục vụ trực tiếp cho trường content_retrieval trong Dual Representation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def is_group_header_row(row: List[str]) -> bool:
    if not row:
        return False
    first_cell = row[0].strip()
    if not first_cell:
        return False
    rest_cells = [c.strip() for c in row[1:]]
    return all(c == "" or c == "-" or c == "—" for c in rest_cells)


def clean_cell_text(cell: str) -> str:
    c = str(cell).replace("\n", " ").strip()
    c = re.sub(r"\s+", " ", c)
    return c


def linearize_financial_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    unit_meta: Dict[str, str],
    period_dates: List[str],
    ticker: str,
    fiscal_year: int,
) -> str:
    """
    Biến đổi bảng thành chuỗi văn bản tự nhiên có cấu trúc phục vụ BM25 và Vector Embedding.
    """
    lines: List[str] = []

    # Breadcrumb header
    lines.append(f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | {caption}]")
    lines.append(f"[Unit: {unit_meta.get('currency', 'USD')} ({unit_meta.get('scale', 'million')})]")
    if period_dates:
        lines.append(f"[Observed Period Dates: {', '.join(period_dates)}]")

    if not rows:
        return "\n".join(lines)

    # Chuẩn hóa headers và giải quyết trường hợp bảng không có header
    clean_headers = [clean_cell_text(h) for h in headers] if headers else []
    
    # 1. Nếu headers bị rỗng hoặc toàn chuỗi trống, kiểm tra xem rows[0] có phải là dòng chứa năm/kỳ không
    effective_rows = rows
    if (not any(clean_headers) or len(clean_headers) <= 1) and rows:
        first_row_non_first = [clean_cell_text(c) for c in rows[0][1:]]
        # Nếu dòng đầu chứa năm (vd: 2024, 2023, FY25) hoặc kỳ báo cáo
        has_year_in_first_row = any(re.search(r"\b(?:FY)?(19|20)\d{2}\b", c, re.I) for c in first_row_non_first if c)
        if has_year_in_first_row:
            clean_headers = [clean_cell_text(c) for c in rows[0]]
            effective_rows = rows[1:]

    # 2. Xây dựng nhãn cột (col_labels) với fallback thông minh theo period_dates / fiscal_year
    col_labels: List[str] = []
    num_cols = len(effective_rows[0]) - 1 if effective_rows and len(effective_rows[0]) > 1 else 0

    for c_idx in range(num_cols):
        # Ưu tiên 1: Header thật từ bảng
        label = ""
        if clean_headers and (c_idx + 1) < len(clean_headers):
            label = clean_headers[c_idx + 1].strip()

        # Ưu tiên 2: Nếu nhãn rỗng hoặc dạng generic (Col_1), sử dụng period_dates quan sát được
        if not label or re.match(r"^col_?\d+$", label, re.I):
            if period_dates and c_idx < len(period_dates):
                label = period_dates[c_idx]
            else:
                # Ưu tiên 3: Suy luận theo niên độ tài chính giảm dần (FY2025, FY2024...)
                label = f"FY{fiscal_year - c_idx}"

        col_labels.append(label)

    rows = effective_rows

    current_parent_label: Optional[str] = None

    for row in rows:
        if not row:
            continue

        raw_label = clean_cell_text(row[0])
        if not raw_label:
            continue

        # Kiểm tra xem có phải dòng tiêu đề nhóm (Group Header) không
        if is_group_header_row(row):
            current_parent_label = raw_label
            lines.append(f"\n--- Category: {current_parent_label} ---")
            continue

        # Kiểm tra reset khi gặp dòng Total
        is_total_row = bool(re.match(r"^total\b", raw_label, re.I))

        # Phân cấp cha - con
        if current_parent_label and not is_total_row:
            full_item_label = f"{current_parent_label} > {raw_label}"
        else:
            full_item_label = raw_label
            if is_total_row:
                current_parent_label = None

        # Ghép các giá trị theo từng cột năm
        val_parts: List[str] = []
        for c_idx, val in enumerate(row[1:]):
            clean_val = clean_cell_text(val)
            if not clean_val or clean_val in ("-", "—"):
                continue

            col_name = col_labels[c_idx] if c_idx < len(col_labels) else f"Period_{c_idx+1}"
            val_parts.append(f"{col_name}: {clean_val}")

        if val_parts:
            lines.append(f"- Line item: {full_item_label} | {', '.join(val_parts)}")

    return "\n".join(lines)
