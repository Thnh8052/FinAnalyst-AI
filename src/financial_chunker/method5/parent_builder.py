"""
parent_builder.py - Method 5 Multi-Tier Parent Chunks and Document-Level Specialized Chunks (Tier 0)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from financial_chunker.models import ChunkMethod, ChunkType, FinancialChunk
from financial_chunker.method5.common import (
    MAX_PARENT_TOKENS,
    render_markdown_table,
    StructuralUnit,
)
from financial_chunker.method5.headings import (
    is_major_subsection,
    is_noise_or_fragment,
    is_top_level_heading,
)


def build_parent_chunks(
    units: List[StructuralUnit],
    child_chunks: List[FinancialChunk],
    ticker: str,
    fiscal_year: int,
    dir_name: str,
    enc: tiktoken.Encoding,
) -> Tuple[List[FinancialChunk], List[FinancialChunk]]:
    """
    Xây dựng các Parent Chunks từ Structural Units và thiết lập liên kết 1-N 2 chiều
    giữa Parent Chunk và Child Chunks:
    - Ranh giới Parent Chunk: Bám sát theo Section/Note (không vượt qua Top-Level Heading).
    - Ngưỡng kích thước: 800 - 2,200 tokens (MAX_PARENT_TOKENS). Nếu một Note vượt quá 2,200 tokens,
      tách theo major subsection hoặc sub-parts tại ranh giới paragraph/table.
    - Tuyệt đối không cắt ngang bảng số liệu (Table Atomicity).
    - Ánh xạ 2 chiều:
      * Child.parent_chunk_id -> Parent.chunk_id
      * Parent.child_chunk_ids -> [Child.chunk_id, ...]
    """
    if not units:
        return [], child_chunks

    parent_groups: List[List[StructuralUnit]] = []
    curr_group: List[StructuralUnit] = []
    curr_tokens = 0
    curr_top_level: Optional[str] = None
    curr_major_sub: Optional[str] = None

    for u in units:
        h_path = getattr(u, "heading_path", [])
        top_h = h_path[0] if h_path else (u.heading_context or "")
        sub_h = h_path[1] if len(h_path) > 1 else ""

        is_new_top = (
            (curr_top_level is not None and top_h != curr_top_level) or
            (u.unit_type == "H" and is_top_level_heading(u.text))
        )
        is_year_h = (u.unit_type == "H" and bool(re.match(r"^(?:19|20)\d{2}\s+", u.text.strip())))
        curr_has_year = any(re.match(r"^(?:19|20)\d{2}\s+", x.text.strip()) for x in curr_group if x.unit_type == "H")

        is_new_major_sub = (
            (is_year_h and curr_has_year) or
            (curr_tokens >= 1200 and (
                (u.unit_type == "H" and is_major_subsection(u.text)) or
                (sub_h and curr_major_sub and sub_h != curr_major_sub and is_major_subsection(sub_h))
            ))
        )
        is_overflow = (curr_tokens + u.tokens > MAX_PARENT_TOKENS and curr_tokens >= 600)

        if curr_group and (is_new_top or is_new_major_sub or is_overflow):
            parent_groups.append(curr_group)
            curr_group = []
            curr_tokens = 0

        curr_group.append(u)
        curr_tokens += u.tokens
        curr_top_level = top_h
        if sub_h and is_major_subsection(sub_h):
            curr_major_sub = sub_h

    if curr_group:
        parent_groups.append(curr_group)

    # Xây dựng các đối tượng FinancialChunk cho Parent
    parent_chunks: List[FinancialChunk] = []
    parent_table_map: Dict[str, FinancialChunk] = {}
    unit_to_parent: Dict[str, FinancialChunk] = {}

    for p_idx, p_group in enumerate(parent_groups, 1):
        first_u = p_group[0]
        h_path = getattr(first_u, "heading_path", [])
        top_title = h_path[0] if h_path else (first_u.heading_context or f"{ticker} FY{fiscal_year} 10-K Section")
        sub_title = h_path[1] if len(h_path) > 1 else ""
        parent_title = f"{top_title} — {sub_title}" if (sub_title and sub_title not in top_title) else top_title

        header_line = f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | Section: {parent_title}]"
        body_parts = []

        seen_headings = set()
        for u in p_group:
            if u.unit_type == "H":
                h_text = u.text.strip()
                if h_text and not is_noise_or_fragment(h_text) and h_text not in seen_headings:
                    seen_headings.add(h_text)
                    lvl = min(len(getattr(u, "heading_path", [])) + 1, 4)
                    prefix = "#" * max(lvl, 2)
                    body_parts.append(f"{prefix} {h_text}")
            elif u.unit_type == "P":
                p_text = u.text.strip()
                if p_text and not is_noise_or_fragment(p_text):
                    body_parts.append(p_text)
            elif u.unit_type == "T":
                if u.table_obj:
                    t = u.table_obj
                    t_cap = (t.caption or "").strip()
                    clean_cap = t_cap.split(" > ")[-1].strip() if " > " in t_cap else t_cap
                    GENERIC_CAPTIONS = {"", "General Financial Information", "Table", "table", "Financial Information"}
                    if clean_cap and clean_cap not in GENERIC_CAPTIONS and not clean_cap.startswith("NOTE"):
                        body_parts.append(f"#### Table: {clean_cap}")
                    md_t = render_markdown_table(t.headers, t.rows)
                    if md_t:
                        body_parts.append(md_t)
                    if t.footnotes:
                        body_parts.append("Footnotes:\n" + "\n".join(t.footnotes))
                elif u.text.strip():
                    body_parts.append(u.text.strip())
            elif u.unit_type == "FN":
                fn_text = u.text.strip()
                if fn_text:
                    body_parts.append(f"Footnotes:\n{fn_text}")

        parent_body = "\n\n".join(body_parts).strip()
        full_parent_content = f"{header_line}\n\n{parent_body}"
        p_tokens = len(enc.encode(full_parent_content))
        p_pages = sorted(list(set(
            p for u in p_group for p in (
                u.table_obj.source_pages if (getattr(u, "table_obj", None) and u.table_obj.source_pages) else [u.page]
            )
        )))

        parent_id = f"{ticker.lower()}_{fiscal_year}_m5_parent_{p_idx:04d}"
        p_chunk = FinancialChunk(
            chunk_id=parent_id,
            document_id=dir_name,
            ticker=ticker,
            fiscal_year=fiscal_year,
            chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
            chunk_type=ChunkType.PARENT_SECTION.value,
            chunk_level="parent",
            parent_chunk_id=None,
            child_chunk_ids=[],
            content=full_parent_content,
            content_retrieval=full_parent_content,
            content_generation=full_parent_content,
            token_count=p_tokens,
            source_pages=p_pages,
            has_table_fragmentation=False,
            metadata={
                "section_title": parent_title,
                "section_hierarchy": list(h_path) if h_path else [parent_title],
                "source_pages": p_pages,
                "total_units": len(p_group),
            },
        )
        parent_chunks.append(p_chunk)

        # Gán mapping
        for u in p_group:
            unit_to_parent[u.unit_id] = p_chunk
            if u.table_obj and u.table_obj.logical_table_id:
                parent_table_map[u.table_obj.logical_table_id] = p_chunk

    # Bước 2: Thiết lập liên kết 1-N 2 chiều giữa Parent và Child
    for child in child_chunks:
        if getattr(child, "chunk_level", None) == "tier0":
            continue

        child.chunk_level = "child"
        target_parent: Optional[FinancialChunk] = None

        # 1. Khớp qua logical_table_id nếu là bảng
        if child.chunk_type in (ChunkType.TABLE_ATOMIC.value, ChunkType.TABLE_SUBGROUP.value):
            tbl_id = child.metadata.get("logical_table_id")
            if tbl_id and tbl_id in parent_table_map:
                target_parent = parent_table_map[tbl_id]

        # 2. Khớp qua section hierarchy (ưu tiên độ sâu trùng khớp) và source_pages
        if target_parent is None:
            child_hier = child.metadata.get("section_hierarchy", [])
            child_pages = set(child.source_pages)
            best_score = -1

            for p in parent_chunks:
                p_hier = p.metadata.get("section_hierarchy", [])
                p_pages = set(p.source_pages)

                # Đếm số cấp tiêu đề trùng khớp từ gốc xuống ngọn
                level_matches = 0
                for ch_h, pr_h in zip(child_hier, p_hier):
                    if ch_h.strip().lower() == pr_h.strip().lower():
                        level_matches += 1
                    else:
                        break

                overlap_pages = len(child_pages.intersection(p_pages))
                score = (level_matches * 100) + overlap_pages
                if score > best_score:
                    best_score = score
                    target_parent = p

        # 3. Gán liên kết 2 chiều
        if target_parent is not None:
            child.parent_chunk_id = target_parent.chunk_id
            if child.chunk_id not in target_parent.child_chunk_ids:
                target_parent.child_chunk_ids.append(child.chunk_id)

    return parent_chunks, child_chunks


def create_specialized_financial_chunks(
    pages_data: List[Dict[str, Any]],
    ticker: str,
    fiscal_year: int,
    dir_name: str,
    enc: tiktoken.Encoding,
) -> List[FinancialChunk]:
    """
    Tạo các chunk chuyên biệt cấp tài liệu (Tier 0):
    1. Cross-Reference Index (Danh mục Thuyết minh báo cáo tài chính).
    2. Document Summary (Tổng quan tài chính điều hành).
    3. Table of Contents (Mục lục cấu trúc Form 10-K).
    """
    specialized_chunks: List[FinancialChunk] = []

    # 1. Cross-Reference Index
    notes_map: Dict[str, Dict[str, Any]] = {}
    for p_data in pages_data:
        p_obj = p_data.get("page", {})
        pdf_page = p_obj.get("pdf_page", 1)
        blocks = p_obj.get("blocks", [])
        for b in blocks:
            if b.get("block_type") == "heading":
                h_text = b.get("text", "").strip()
                m = re.match(r"(?:Note|Thuyết\s+minh)\s*(\d+[A-Za-z]?)\s*[-:–—.]\s*(.*)", h_text, re.I)
                if m:
                    n_code = m.group(1).upper()
                    n_title = m.group(2).strip()
                    if n_code not in notes_map:
                        notes_map[n_code] = {"title": n_title, "first_page": pdf_page}

    if notes_map:
        idx_lines = [
            f"# {ticker} Corporation FY{fiscal_year} 10-K Notes to Financial Statements Directory",
            f"Document: {ticker} FY{fiscal_year} | Master Cross-Reference Navigation Index\n",
            "| Note Number | Title / Accounting Subject | Start Page |",
            "| :--- | :--- | :--- |",
        ]
        sorted_notes = sorted(
            notes_map.items(),
            key=lambda item: int(re.match(r"^\d+", item[0]).group(0)) if re.match(r"^\d+", item[0]) else 999,
        )
        for n_code, n_info in sorted_notes:
            idx_lines.append(f"| Note {n_code} | {n_info['title']} | Page {n_info['first_page']} |")

        index_text = "\n".join(idx_lines)
        index_tokens = len(enc.encode(index_text))
        all_pages = sorted(list({n["first_page"] for n in notes_map.values()}))

        specialized_chunks.append(
            FinancialChunk(
                chunk_id=f"{ticker.lower()}_{fiscal_year}_m5_xref_index",
                document_id=dir_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
                chunk_type=ChunkType.CROSS_REFERENCE_INDEX.value,
                chunk_level="tier0",
                content=index_text,
                content_retrieval=index_text,
                content_generation=index_text,
                token_count=index_tokens,
                source_pages=all_pages[:10],
                has_table_fragmentation=False,
                metadata={"total_notes_indexed": len(notes_map)},
            )
        )

    # 2. Document Summary
    summary_lines = [
        f"# {ticker} Corporation FY{fiscal_year} Annual Report (Form 10-K) Key Financial Highlights",
        f"[Company: {ticker}] [Fiscal Year: {fiscal_year}] [Filing Type: Form 10-K]",
        "\nCore Financial Information Overview:",
        f"- Ticker: {ticker}",
        f"- Fiscal Year Ended: {fiscal_year}",
        "- Reporting Standard: U.S. GAAP",
        "- Audited Status: Audited Consolidated Financial Statements",
    ]
    summary_text = "\n".join(summary_lines)
    s_tokens = len(enc.encode(summary_text))

    specialized_chunks.append(
        FinancialChunk(
            chunk_id=f"{ticker.lower()}_{fiscal_year}_m5_doc_summary",
            document_id=dir_name,
            ticker=ticker,
            fiscal_year=fiscal_year,
            chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
            chunk_type=ChunkType.DOCUMENT_SUMMARY.value,
            chunk_level="tier0",
            content=summary_text,
            content_retrieval=summary_text,
            content_generation=summary_text,
            token_count=s_tokens,
            source_pages=[1],
            has_table_fragmentation=False,
            metadata={"summary_scope": "executive_financial_overview"},
        )
    )

    # 3. Table of Contents
    toc_items: List[Dict[str, Any]] = []
    seen_toc: set = set()
    for p_data in pages_data:
        p_obj = p_data.get("page", {})
        pdf_page = p_obj.get("pdf_page", 1)
        blocks = p_obj.get("blocks", [])
        for b in blocks:
            if b.get("block_type") == "heading":
                h_text = b.get("text", "").strip()
                s_up = h_text.upper()
                if (
                    re.match(r"^PART\s+[IVX]+", s_up) or
                    re.match(r"^ITEM\s+\d+[A-Z]?", s_up) or
                    "CONSOLIDATED STATEMENTS" in s_up or
                    "CONSOLIDATED BALANCE SHEETS" in s_up
                ):
                    if h_text not in seen_toc:
                        seen_toc.add(h_text)
                        toc_items.append({"title": h_text, "page": pdf_page})

    if toc_items:
        toc_lines = [
            f"# {ticker} Corporation FY{fiscal_year} Form 10-K Table of Contents",
            f"Document: {ticker} FY{fiscal_year} | Master Table of Contents\n",
            "| Section / Item | Start Page |",
            "| :--- | :--- |",
        ]
        for item in toc_items:
            toc_lines.append(f"| {item['title']} | Page {item['page']} |")
        toc_text = "\n".join(toc_lines)
        toc_tokens = len(enc.encode(toc_text))
        all_pages = sorted(list({item["page"] for item in toc_items}))
        specialized_chunks.append(
            FinancialChunk(
                chunk_id=f"{ticker.lower()}_{fiscal_year}_m5_toc",
                document_id=dir_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
                chunk_type=ChunkType.TABLE_OF_CONTENTS.value,
                chunk_level="tier0",
                content=toc_text,
                content_retrieval=toc_text,
                content_generation=toc_text,
                token_count=toc_tokens,
                source_pages=all_pages[:10],
                has_table_fragmentation=False,
                metadata={"total_items_indexed": len(toc_items)},
            )
        )

    return specialized_chunks
