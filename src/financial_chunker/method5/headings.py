"""
headings.py - Method 5 Heading Hierarchy, Breadcrumb Stack, and Noise Filtering
"""

from __future__ import annotations

import re
from typing import List


def is_top_level_heading(h: str) -> bool:
    """Nhận diện tiêu đề cấp cao nhất (Note, Item, Part, BCTC chính)."""
    s = h.strip().upper()
    return bool(
        re.match(r"^(?:NOTE|THUYẾT\s+MINH)\s*\d+", s) or
        re.match(r"^(?:ITEM|MỤC)\s*\d+[A-Z]?", s) or
        re.match(r"^PART\s+[IVX]+", s) or
        re.match(r"^FORM\s+10-[KQ]", s) or
        "CONSOLIDATED STATEMENTS" in s or
        "CONSOLIDATED BALANCE SHEETS" in s
    )


def is_major_subsection(h: str) -> bool:
    """Nhận diện tiêu đề tiểu mục lớn độc lập (Năm, Divestiture, Goodwill, Segment...)."""
    s = h.strip()
    return bool(
        re.match(r"^(?:19|20)\d{2}\s+", s) or
        re.match(r"^(?:Divestiture|Acquisition|Goodwill|Other\s+Restructuring|Segment|Subsequent|Accounting\s+Policies)", s, re.I)
    )


def update_heading_stack(heading_stack: List[str], new_heading: str) -> List[str]:
    """Cập nhật ngăn xếp tiêu đề phân cấp 3 tầng chuẩn 10-K."""
    clean_h = new_heading.strip()
    if not clean_h:
        return list(heading_stack)
    
    clean_h = re.sub(r"^#+\s*", "", clean_h).strip()

    if is_top_level_heading(clean_h):
        return [clean_h]

    if not heading_stack:
        return [clean_h]

    if len(heading_stack) == 1:
        return [heading_stack[0], clean_h]

    curr_l2_is_year = bool(re.match(r"^(?:19|20)\d{2}\s+", heading_stack[1]))
    new_is_year = bool(re.match(r"^(?:19|20)\d{2}\s+", clean_h))
    new_is_major = is_major_subsection(clean_h)

    if len(heading_stack) == 2:
        # Nếu heading mới là chương trình/chủ đề lớn độc lập (Divestiture, Năm mới, Goodwill, Segment...)
        # thì thay thế cấp 2 (sibling)
        if new_is_major and (new_is_year or not curr_l2_is_year or re.match(r"^(?:Divestiture|Acquisition|Goodwill|Other\s+Restructuring)", clean_h, re.I)):
            return [heading_stack[0], clean_h]
        else:
            # Ngược lại, là tiểu mục con (Operational/Marketing...) đẩy lên cấp 3
            return [heading_stack[0], heading_stack[1], clean_h]

    if len(heading_stack) >= 3:
        if new_is_major:
            # Chuyển sang major section mới -> reset cấp 3 về cấp 2
            return [heading_stack[0], clean_h]
        else:
            # Tiểu mục ngang hàng cấp 3
            return [heading_stack[0], heading_stack[1], clean_h]

    return [heading_stack[0], clean_h]


def is_noise_or_fragment(text: str) -> bool:
    """Nhận diện và loại bỏ rác/mẩu chữ cụt từ PDF parsing."""
    s = text.strip()
    if not s:
        return True
    if s in (":", "-", "--", "---"):
        return True
    # Mẩu chữ cụt (< 35 ký tự) kết thúc bằng liên từ/giới từ lơ lửng từ PDF parsing
    if len(s) < 35 and re.search(r"\b(?:and|or|of|to|the|with|in|for|as|follow|follows)\s*[:.]?$", s, re.I):
        return True
    # Số trang hoặc ký tự lẻ
    if re.match(r"^\d{1,4}$", s):
        return True
    return False
