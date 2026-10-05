"""
consolidator.py - Method 5 Structural Unit Extraction and Semantic Group Consolidation
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import tiktoken

from financial_chunker.models import StitchedDocument, StitchedTable
from financial_chunker.method5.common import (
    FOOTNOTE_PATTERN,
    render_markdown_table,
    StructuralUnit,
    TARGET_GROUP_MAX_TOKENS,
)
from financial_chunker.method5.headings import (
    is_noise_or_fragment,
    is_top_level_heading,
    update_heading_stack,
)


def extract_structural_units(
    pages_data: List[Dict[str, Any]],
    stitched_doc: StitchedDocument,
    enc: tiktoken.Encoding,
) -> List[StructuralUnit]:
    """
    Trích xuất danh sách Structural Units tuần tự từ tài liệu:
    - Bảng đa trang đã được SectionStitcher ghép nối, chỉ xuất hiện một lần tại vị trí ban đầu.
    - Footnote được phân loại riêng để LLM có thể quyết định gộp kèm bảng cha.
    """
    # Xây dựng bản đồ tra cứu stitched tables theo table_id và block_id
    stitched_table_map: Dict[str, StitchedTable] = {}
    stitched_seen_tables: set = set()

    for st in stitched_doc.tables:
        for tid in st.source_table_ids:
            stitched_table_map[tid] = st
        for bid in st.source_block_ids:
            stitched_table_map[bid] = st

    sorted_pages = sorted(
        pages_data,
        key=lambda p: p.get("page", {}).get("pdf_page", 1),
    )

    units: List[StructuralUnit] = []
    unit_by_id: Dict[str, StructuralUnit] = {}
    unit_counter = 0
    heading_stack: List[str] = ["General Financial Information"]
    current_heading = "General Financial Information"
    last_table_unit_id: Optional[str] = None

    for p_data in sorted_pages:
        page_obj = p_data.get("page", {})
        pdf_page = page_obj.get("pdf_page", 1)
        page_qc = page_obj.get("qc", {}).get("status", "pass")
        blocks = page_obj.get("blocks", [])

        for b_idx, block in enumerate(blocks):
            b_type = block.get("block_type")
            b_id = block.get("block_id", f"p{pdf_page}_b{b_idx}")

            if b_type == "heading":
                h_text = block.get("text", "").strip()
                if not h_text:
                    continue
                if is_top_level_heading(h_text):
                    last_table_unit_id = None
                heading_stack = update_heading_stack(heading_stack, h_text)
                current_heading = " > ".join(heading_stack)
                unit_counter += 1
                tokens = len(enc.encode(h_text))
                h_unit = StructuralUnit(
                    unit_id=f"u{unit_counter:04d}",
                    unit_type="H",
                    page=pdf_page,
                    text=h_text,
                    tokens=tokens,
                    preview=f"Heading: {h_text[:70]}",
                    heading_context=current_heading,
                    heading_path=list(heading_stack),
                    qc_status=page_qc,
                )
                units.append(h_unit)
                unit_by_id[h_unit.unit_id] = h_unit

            elif b_type == "table":
                t_id = block.get("table_id", b_id)
                st_obj = stitched_table_map.get(t_id) or stitched_table_map.get(b_id)

                if st_obj:
                    if st_obj.logical_table_id in stitched_seen_tables:
                        # Bảng đa trang đã được thêm từ trang trước -> bỏ qua block continuation
                        continue
                    stitched_seen_tables.add(st_obj.logical_table_id)
                    table_to_use = st_obj
                else:
                    # Bảng đơn trang độc lập
                    table_to_use = StitchedTable(
                        logical_table_id=f"single_{t_id}",
                        source_pages=[pdf_page],
                        source_table_ids=[t_id],
                        source_block_ids=[b_id],
                        headers=block.get("headers", []),
                        rows=block.get("rows", []),
                        caption=block.get("caption") or block.get("text") or current_heading,
                        unit=block.get("unit") or page_obj.get("unit"),
                        period=block.get("period") or page_obj.get("period"),
                        footnotes=[],
                        qc_status=page_qc,
                    )

                caption = table_to_use.caption or current_heading
                md_repr = render_markdown_table(table_to_use.headers, table_to_use.rows)
                t_tokens = len(enc.encode(md_repr))

                unit_counter += 1
                u_id = f"u{unit_counter:04d}"
                last_table_unit_id = u_id

                t_unit = StructuralUnit(
                    unit_id=u_id,
                    unit_type="T",
                    page=pdf_page,
                    text=md_repr,
                    tokens=t_tokens,
                    preview=f"Table: {caption[:60]} ({len(table_to_use.rows)} rows)",
                    heading_context=current_heading,
                    heading_path=list(heading_stack),
                    table_obj=table_to_use,
                    qc_status=table_to_use.qc_status,
                )
                units.append(t_unit)
                unit_by_id[t_unit.unit_id] = t_unit

            elif b_type == "paragraph":
                p_text = block.get("text", "").strip()
                if not p_text or is_noise_or_fragment(p_text):
                    continue

                p_tokens = len(enc.encode(p_text))

                # Kiểm tra xem có phải footnote của bảng liền trước trong cùng section không
                is_fn = bool(FOOTNOTE_PATTERN.match(p_text))
                tbl_u = unit_by_id.get(last_table_unit_id) if last_table_unit_id else None
                same_section = (
                    tbl_u is not None and (
                        tbl_u.heading_context == current_heading or
                        (bool(tbl_u.heading_path) and bool(heading_stack) and tbl_u.heading_path[0] == heading_stack[0])
                    )
                )
                u_type = "FN" if (is_fn and same_section) else "P"

                unit_counter += 1
                p_unit = StructuralUnit(
                    unit_id=f"u{unit_counter:04d}",
                    unit_type=u_type,
                    page=pdf_page,
                    text=p_text,
                    tokens=p_tokens,
                    preview=f"{u_type}: {p_text[:75]}",
                    heading_context=current_heading,
                    heading_path=list(heading_stack),
                    qc_status=page_qc,
                )
                units.append(p_unit)
                unit_by_id[p_unit.unit_id] = p_unit

    return units


def consolidate_semantic_groups(
    raw_groups: List[List[str]],
    unit_map: Dict[str, StructuralUnit],
    enc: Optional[tiktoken.Encoding] = None,
) -> List[List[str]]:
    """
    Semantic & Layout Consolidation Engine:
    1. Tách các bảng (Atomic Table Splitting): Mỗi bảng số liệu T là một đơn vị độc lập.
    2. Triệt tiêu các nhóm tiêu đề mồ côi (Absorb Orphan Heading Groups):
       Các nhóm chỉ gồm unit_type == 'H' không được phép tồn tại độc lập,
       mà được chuyển tiếp (prepend) vào nhóm nội dung liền sau.
    3. Gắn kết câu kết luận / ghi chú sau bảng (Post-table commentary binding):
       Đoạn văn ngắn sau bảng (< 150 tokens) được gắn vào bảng trước làm Post-Table Notes.
    4. Gom cụm đoạn văn thuyết minh (Narrative Consolidation):
       Các nhóm văn bản ngắn (< 120 tokens) được gộp với văn bản kế tiếp trong cùng section,
       loại bỏ hoàn toàn các chunk 20-50 tokens ngắn cụt.
    """
    if not raw_groups:
        return []

    # BƯỚC 1: Tách các bảng đa bảng trong cùng một nhóm (Atomic Table Splitting)
    stage1_groups: List[List[str]] = []
    for g_ids in raw_groups:
        g_units = [unit_map[uid] for uid in g_ids if uid in unit_map]
        t_indices = [i for i, u in enumerate(g_units) if u.unit_type == "T"]
        if len(t_indices) <= 1:
            stage1_groups.append(g_ids)
        else:
            sub_g: List[str] = []
            for u in g_units:
                if (u.unit_type in ("T", "H")) and any(unit_map[x].unit_type == "T" for x in sub_g):
                    if sub_g:
                        stage1_groups.append(sub_g)
                    sub_g = [u.unit_id]
                elif u.unit_type == "P" and any(unit_map[x].unit_type == "T" for x in sub_g):
                    p_txt = u.text.strip()
                    if p_txt.endswith(":") or re.search(r"(?:follow|follows|summarized as follows|as follows)\s*:\s*$", p_txt, re.I):
                        if sub_g:
                            stage1_groups.append(sub_g)
                        sub_g = [u.unit_id]
                    else:
                        sub_g.append(u.unit_id)
                else:
                    sub_g.append(u.unit_id)
            if sub_g:
                stage1_groups.append(sub_g)

    # BƯỚC 2: Triệt tiêu các nhóm tiêu đề mồ côi (Absorb Orphan Heading Groups)
    stage2_groups: List[List[str]] = []
    pending_headings: List[str] = []

    for g_ids in stage1_groups:
        g_units = [unit_map[uid] for uid in g_ids if uid in unit_map]
        is_only_headings = all(u.unit_type == "H" for u in g_units) if g_units else True

        if is_only_headings:
            pending_headings.extend(g_ids)
            continue

        if pending_headings:
            stage2_groups.append(pending_headings + g_ids)
            pending_headings = []
        else:
            stage2_groups.append(g_ids)

    if pending_headings:
        if stage2_groups:
            stage2_groups[-1].extend(pending_headings)
        else:
            stage2_groups.append(pending_headings)

    # BƯỚC 3: Gắn kết câu kết luận / ghi chú sau bảng (Post-table commentary binding)
    stage3_groups: List[List[str]] = []
    idx = 0
    while idx < len(stage2_groups):
        curr_g = list(stage2_groups[idx])
        curr_units = [unit_map[uid] for uid in curr_g if uid in unit_map]
        has_table = any(u.unit_type == "T" for u in curr_units)

        if has_table and idx + 1 < len(stage2_groups):
            next_g = stage2_groups[idx + 1]
            next_units = [unit_map[uid] for uid in next_g if uid in unit_map]
            if (
                next_units and 
                not any(u.unit_type in ("T", "H") for u in next_units) and
                sum(u.tokens for u in next_units) < 120
            ):
                first_next_p = next_units[0].text.strip()
                if not first_next_p.endswith(":") and not re.search(r"(?:follow|follows|as follows)\s*:\s*$", first_next_p, re.I):
                    combined_toks = sum(u.tokens for u in curr_units) + sum(u.tokens for u in next_units)
                    if combined_toks <= TARGET_GROUP_MAX_TOKENS:
                        curr_g.extend(next_g)
                        idx += 1

        stage3_groups.append(curr_g)
        idx += 1

    # BƯỚC 4: Gom cụm các nhóm thuyết minh ngắn (Narrative Consolidation < 120 tokens)
    final_groups: List[List[str]] = []
    accum_narrative: List[str] = []
    accum_tokens = 0

    for g_ids in stage3_groups:
        g_units = [unit_map[uid] for uid in g_ids if uid in unit_map]
        has_table = any(u.unit_type == "T" for u in g_units)

        if has_table:
            if accum_narrative:
                final_groups.append(accum_narrative)
                accum_narrative = []
                accum_tokens = 0
            final_groups.append(g_ids)
        else:
            curr_top_sec = g_units[0].heading_path[0] if (g_units and getattr(g_units[0], "heading_path", None) and g_units[0].heading_path) else ""
            accum_first = unit_map.get(accum_narrative[0]) if accum_narrative else None
            accum_top_sec = (
                accum_first.heading_path[0]
                if (accum_first and getattr(accum_first, "heading_path", None) and accum_first.heading_path)
                else ""
            )

            # Rào chắn section: Nếu chuyển sang Note hoặc Item mới, bắt buộc flush narrative của Note trước!
            if accum_narrative and curr_top_sec and accum_top_sec and curr_top_sec != accum_top_sec:
                final_groups.append(accum_narrative)
                accum_narrative = []
                accum_tokens = 0

            g_tokens = sum(u.tokens for u in g_units)
            if accum_tokens + g_tokens <= TARGET_GROUP_MAX_TOKENS:
                accum_narrative.extend(g_ids)
                accum_tokens += g_tokens
                if accum_tokens >= 400:
                    final_groups.append(accum_narrative)
                    accum_narrative = []
                    accum_tokens = 0
            else:
                if accum_narrative:
                    final_groups.append(accum_narrative)
                accum_narrative = list(g_ids)
                accum_tokens = g_tokens

    if accum_narrative:
        if final_groups and accum_tokens < 100:
            prev_units = [unit_map[uid] for uid in final_groups[-1] if uid in unit_map]
            if not any(u.unit_type == "T" for u in prev_units):
                prev_top = prev_units[0].heading_path[0] if (prev_units and getattr(prev_units[0], "heading_path", None) and prev_units[0].heading_path) else ""
                accum_first = unit_map.get(accum_narrative[0]) if accum_narrative else None
                accum_top = (
                    accum_first.heading_path[0]
                    if (accum_first and getattr(accum_first, "heading_path", None) and accum_first.heading_path)
                    else ""
                )
                if prev_top == accum_top:
                    prev_toks = sum(u.tokens for u in prev_units)
                    if prev_toks + accum_tokens <= TARGET_GROUP_MAX_TOKENS:
                        final_groups[-1].extend(accum_narrative)
                        accum_narrative = []
        if accum_narrative:
            final_groups.append(accum_narrative)

    return final_groups
