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

from typing import Dict, List, Optional


def clean_cell_text(cell: str) -> str:
    """A1: Chuẩn hoá whitespace siêu tốc bằng split(), không dùng regex."""
    return " ".join(str(cell).split())


def _starts_with_word(s: str, word: str) -> bool:
    """A6: Kiểm tra từ đứng đầu theo ranh giới từ (word boundary) không dùng regex."""
    s_clean = s.strip().lower()
    if not s_clean.startswith(word):
        return False
    if len(s_clean) == len(word):
        return True
    return not s_clean[len(word)].isalnum()


def _is_generic_col_label(label: str) -> bool:
    """A7: Nhận diện nhãn cột generic (col_1, col2, column1, column_2) không dùng regex."""
    s = label.strip().lower()
    for prefix in ("column", "col"):
        if s.startswith(prefix):
            rest = s[len(prefix):].lstrip("_").strip()
            return len(rest) > 0 and rest.isdigit()
    return False


def is_group_header_row(row: List[str]) -> bool:
    """Bug L1: Nhận diện hàng header nhóm, chặn hàng đơn ô (<= 1 cell)."""
    if not row or len(row) <= 1:
        return False
    first_cell = clean_cell_text(row[0])
    if not first_cell:
        return False
    EMPTY_CELLS = {"", "-", "—", "–", "n/a", "N/A", "$ -", "$-", "$—"}
    return all(clean_cell_text(c) in EMPTY_CELLS for c in row[1:])


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
        has_year_in_first_row = any(
            any(t.isdigit() and len(t) == 4 and t.startswith(("19", "20")) for t in c.replace("FY", "").split())
            for c in first_row_non_first if c
        )
        if has_year_in_first_row:
            clean_headers = [clean_cell_text(c) for c in rows[0]]
            effective_rows = rows[1:]

    # Bug #3: Sắp xếp period_dates GIẢM DẦN để khớp thứ tự 10-K (current year first: 2024 -> 2023 -> 2022)
    desc_dates = sorted(period_dates, reverse=True) if period_dates else []

    # Bug L2: Bằng chứng cột thời gian: desc_dates có phần tử HOẶC header có chứa số năm 19xx/20xx
    has_temporal_headers = any(
        any(token.isdigit() and len(token) == 4 and token.startswith(("19", "20"))
            for token in clean_cell_text(h).split())
        for h in clean_headers
    ) or bool(desc_dates)

    # 2. Xây dựng nhãn cột (col_labels) với fallback thông minh
    col_labels: List[str] = []
    num_cols = len(effective_rows[0]) - 1 if effective_rows and len(effective_rows[0]) > 1 else 0

    for c_idx in range(num_cols):
        # Ưu tiên 1: Header thật từ bảng
        label = ""
        if clean_headers and (c_idx + 1) < len(clean_headers):
            label = clean_headers[c_idx + 1].strip()

        # Ưu tiên 2: Nếu nhãn rỗng hoặc dạng generic (Col_1), sử dụng period_dates quan sát được
        if not label or _is_generic_col_label(label):
            if desc_dates and c_idx < len(desc_dates):
                label = desc_dates[c_idx]
            elif has_temporal_headers:
                # Ưu tiên 3: Suy luận theo niên độ tài chính giảm dần khi có bằng chứng thời gian
                label = f"FY{fiscal_year - c_idx}"
            else:
                # Ưu tiên 4 (Bug L2): Không bịa năm nếu là bảng danh mục
                label = f"Category_{c_idx + 1}"

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

        # Kiểm tra reset khi gặp dòng Total / Subtotal (A6)
        is_total_row = _starts_with_word(raw_label, "total") or _starts_with_word(raw_label, "subtotal")

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
