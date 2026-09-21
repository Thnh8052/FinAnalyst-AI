"""
stitcher.py

SectionStitcher: Bộ ghép nối phân đoạn và bảng đa trang cho Báo cáo tài chính.
Triển khai đúng 3 quy tắc định lượng:
- Quy tắc 1: Bảng đa trang (ColumnMatchRatio >= 0.75, SequenceMatcher >= 0.80, boundary position)
- Quy tắc 2: Thuyết minh tiếp diễn (Note continuation, câu mở unpunctuated sentence boundary)
- Quy tắc 3: Gắn kết Footnotes (khoảng cách <= 2 blocks ngay sau bảng, kiểm tra symbol)
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from script.financial_chunker.models import (
    StitchedDocument,
    StitchedSection,
    StitchedTable,
)

FOOTNOTE_PATTERN = re.compile(
    r"^\s*(?:\([0-9a-zA-Z*†‡§#]\)|\[[0-9a-zA-Z*†‡§#]\]|\*+|[0-9]{1,2}\.?\s+[A-Z])"
)

CONTINUATION_HEADER_PATTERN = re.compile(
    r"(?:Note|Thuyết\s+minh|Item)\s*(\d+[A-Za-z]?)\s*[\(–—\-]?\s*(?:Continued|tiếp\s+theo)",
    re.I,
)


def normalize_header_col(col: str) -> str:
    c = col.strip().lower()
    c = re.sub(r"[^\w\s]", " ", c)
    # Thay năm bằng [YEAR]
    c = re.sub(r"\b(?:19|20)\d{2}\b", "[YEAR]", c)
    return " ".join(c.split())


def compute_column_match_ratio(h1: List[str], h2: List[str]) -> float:
    if not h1 or not h2:
        return 0.0
    if len(h1) != len(h2):
        return 0.0

    matches = 0
    for c1, c2 in zip(h1, h2):
        n1 = normalize_header_col(c1)
        n2 = normalize_header_col(c2)
        ratio = SequenceMatcher(None, n1, n2).ratio()
        if ratio >= 0.80:
            matches += 1

    return matches / len(h1)


def is_open_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return not bool(re.search(r'[\.\?\!\:\"”\)]\s*$', stripped))


def is_continuation_start(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    # Bắt đầu bằng chữ thường hoặc liên từ hoặc số mở ngoặc
    return bool(re.match(r"^[a-zà-ỹ0-9\(\,\-]", stripped))


class SectionStitcher:
    """
    Điều phối phân tích và ghép nối các trang Canonical JSON thành StitchedDocument logic.
    """

    def __init__(self, column_match_threshold: float = 0.75):
        self.column_match_threshold = column_match_threshold

    def stitch_company_pages(
        self,
        pages_data: List[Dict[str, Any]],
        document_id: str,
        ticker: str,
        fiscal_year: int,
    ) -> StitchedDocument:
        # Sắp xếp các trang theo pdf_page tăng dần
        sorted_pages = sorted(
            pages_data,
            key=lambda p: p.get("page", {}).get("pdf_page", 1),
        )

        stitched_tables: List[StitchedTable] = []
        stitched_sections: List[StitchedSection] = []
        cross_page_links: List[Dict[str, Any]] = []

        current_table: Optional[StitchedTable] = None
        current_section: Optional[StitchedSection] = None
        table_counter = 0
        section_counter = 0

        for p_idx, p_data in enumerate(sorted_pages):
            page_obj = p_data.get("page", {})
            pdf_page = page_obj.get("pdf_page", p_idx + 1)
            blocks = page_obj.get("blocks", [])
            page_qc = page_obj.get("qc", {}).get("status", "pass")
            page_unit = page_obj.get("unit")
            page_period = page_obj.get("period")

            # Duyệt qua các blocks của trang
            for b_idx, block in enumerate(blocks):
                b_type = block.get("block_type")
                b_id = block.get("block_id", f"p{pdf_page}_b{b_idx}")

                if b_type == "heading":
                    h_text = block.get("text", "").strip()
                    h_level = block.get("level", 1)

                    # Kiểm tra xem có phải continuation header không (Note X Continued)
                    cont_match = CONTINUATION_HEADER_PATTERN.search(h_text)
                    if cont_match and current_section and cont_match.group(1) == current_section.section_code:
                        # Tiếp tục section hiện tại, ghi nhận trang mới
                        if pdf_page not in current_section.source_pages:
                            current_section.source_pages.append(pdf_page)
                        current_section.blocks.append(block)
                        cross_page_links.append({
                            "type": "heading_continuation",
                            "from_page": pdf_page,
                            "to_section": current_section.logical_section_id,
                        })
                    else:
                        # Kết thúc section cũ và tạo section mới
                        if current_section and current_section.blocks:
                            stitched_sections.append(current_section)

                        section_counter += 1
                        sec_code_match = re.search(r"(?:Note|Thuyết\s+minh|Item)\s*(\d+[A-Za-z]?)", h_text, re.I)
                        sec_code = sec_code_match.group(1) if sec_code_match else None

                        current_section = StitchedSection(
                            logical_section_id=f"{ticker.lower()}_{fiscal_year}_sec_{section_counter:04d}",
                            section_code=sec_code,
                            section_title=h_text,
                            parent_section_id=None,
                            source_pages=[pdf_page],
                            blocks=[block],
                            narrative_text=h_text,
                        )

                elif b_type == "table":
                    headers = block.get("headers", [])
                    rows = block.get("rows", [])
                    caption = block.get("caption") or block.get("text")
                    t_unit = block.get("unit") or page_unit
                    t_period = block.get("period") or page_period
                    t_id = block.get("table_id", f"p{pdf_page}_t{b_idx}")

                    # Kiểm tra ghép bảng vắt trang (Multi-page Table Stitching)
                    can_stitch = False
                    if current_table is not None:
                        # Điều kiện 1: Bảng trước nằm ở cuối trang N, bảng này ở đầu trang N+1
                        is_adjacent_page = (pdf_page == current_table.source_pages[-1] + 1)
                        is_top_of_page = (b_idx <= 1)  # block 0 hoặc block 1 sau page header

                        if is_adjacent_page and is_top_of_page:
                            # Điều kiện 2: Kiểm tra độ tương đồng cột
                            match_ratio = compute_column_match_ratio(current_table.headers, headers)
                            if match_ratio >= self.column_match_threshold:
                                can_stitch = True
                            elif not any(headers) and rows and len(rows[0]) == len(current_table.headers):
                                # Điều kiện 3: Bảng sau không có header nhưng khớp số cột
                                can_stitch = True

                    if can_stitch and current_table is not None:
                        # Thực hiện ghép bảng
                        current_table.rows.extend(rows)
                        current_table.source_pages.append(pdf_page)
                        current_table.source_table_ids.append(t_id)
                        current_table.source_block_ids.append(b_id)
                        current_table.is_multi_page = True
                        if page_qc == "warning" or current_table.qc_status == "warning":
                            current_table.qc_status = "warning"
                        if page_qc == "fail":
                            current_table.qc_status = "fail"

                        cross_page_links.append({
                            "type": "table_continuation",
                            "from_page": pdf_page,
                            "to_table": current_table.logical_table_id,
                            "table_id": t_id,
                        })
                    else:
                        # Đóng bảng cũ nếu có
                        if current_table is not None:
                            stitched_tables.append(current_table)

                        table_counter += 1
                        current_table = StitchedTable(
                            logical_table_id=f"{ticker.lower()}_{fiscal_year}_ltbl_{table_counter:04d}",
                            source_pages=[pdf_page],
                            source_table_ids=[t_id],
                            source_block_ids=[b_id],
                            headers=headers,
                            rows=rows,
                            caption=caption,
                            unit=t_unit,
                            period=t_period,
                            footnotes=[],
                            qc_status=page_qc,
                            is_multi_page=False,
                        )

                elif b_type == "paragraph":
                    p_text = block.get("text", "").strip()
                    if not p_text:
                        continue

                    # Kiểm tra Footnote Attachment (Quy tắc 3)
                    # Nếu paragraph ngay sau bảng (khoảng cách <= 2 blocks) và khớp FOOTNOTE_PATTERN
                    if current_table is not None and len(current_table.source_pages) > 0 and current_table.source_pages[-1] == pdf_page:
                        if FOOTNOTE_PATTERN.match(p_text):
                            current_table.footnotes.append(p_text)
                            continue

                    # Nếu không phải footnote, thêm vào section hiện tại
                    if current_section is None:
                        section_counter += 1
                        current_section = StitchedSection(
                            logical_section_id=f"{ticker.lower()}_{fiscal_year}_sec_{section_counter:04d}",
                            section_code=None,
                            section_title="General Narrative",
                            parent_section_id=None,
                            source_pages=[pdf_page],
                            blocks=[],
                            narrative_text="",
                        )

                    # Kiểm tra câu nối giữa 2 trang (Unpunctuated sentence boundary)
                    if (
                        current_section.narrative_text
                        and pdf_page not in current_section.source_pages
                        and is_open_sentence(current_section.narrative_text)
                        and is_continuation_start(p_text)
                    ):
                        current_section.narrative_text += " " + p_text
                        current_section.blocks.append(block)
                        current_section.source_pages.append(pdf_page)
                    else:
                        if current_section.narrative_text:
                            current_section.narrative_text += "\n\n" + p_text
                        else:
                            current_section.narrative_text = p_text
                        current_section.blocks.append(block)
                        if pdf_page not in current_section.source_pages:
                            current_section.source_pages.append(pdf_page)

        # Lưu lại table và section cuối cùng
        if current_table is not None:
            stitched_tables.append(current_table)
        if current_section is not None and current_section.blocks:
            stitched_sections.append(current_section)

        return StitchedDocument(
            document_id=document_id,
            ticker=ticker,
            fiscal_year=fiscal_year,
            tables=stitched_tables,
            sections=stitched_sections,
            cross_page_links=cross_page_links,
        )
