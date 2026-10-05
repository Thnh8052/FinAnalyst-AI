"""
child_builder.py - Method 5 Dual-Representation Child Chunk Builder (Tuples + Markdown)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import tiktoken

from financial_chunker.enricher import (
    detect_statement_type,
    extract_iso_dates_from_period,
    parse_unit_metadata,
)
from financial_chunker.linearizer import is_group_header_row, linearize_financial_table
from financial_chunker.models import ChunkMethod, ChunkType, FinancialChunk
from financial_chunker.method5.common import (
    MAX_TABLE_TOKENS,
    NARRATIVE_OVERLAP_TOKENS,
    NARRATIVE_SPLIT_TOKENS,
    render_markdown_table,
    split_sentences,
    StructuralUnit,
)
from financial_chunker.method5.headings import is_noise_or_fragment


def _make_narrative_chunk(
    chunk_id: str,
    document_id: str,
    ticker: str,
    fiscal_year: int,
    part_text: str,
    part_tokens: int,
    main_heading: str,
    sec_hierarchy: List[str],
    source_pages: List[int],
    qc_warning: bool,
    sub_part: Optional[int] = None,
) -> FinancialChunk:
    first_page = source_pages[0] if source_pages else 1
    metadata: Dict[str, Any] = {
        "section_title": main_heading,
        "section_hierarchy": sec_hierarchy,
        "has_linearized_tuples": False,
        "has_qc_warning": qc_warning,
        "source_verification_badge": (
            f"⚠️ Lưu ý kiểm tra nguồn: Văn bản tại Trang {first_page} "
            "có định dạng phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
            if qc_warning else None
        ),
        "pdf_inspector": {
            "pdf_page": first_page,
            "highlight_target": "text_block",
        },
    }
    if sub_part is not None:
        metadata["sub_part"] = sub_part

    return FinancialChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        ticker=ticker,
        fiscal_year=fiscal_year,
        chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
        chunk_type=ChunkType.NARRATIVE_SECTION.value,
        chunk_level="child",
        content=part_text,
        content_retrieval=part_text,
        content_generation=part_text,
        token_count=part_tokens,
        source_pages=source_pages,
        has_table_fragmentation=False,
        metadata=metadata,
    )


def build_dual_representation_chunk(
    group_units: List[StructuralUnit],
    ticker: str,
    fiscal_year: int,
    dir_name: str,
    chunk_index: int,
    enc: tiktoken.Encoding,
    fiscal_date_suffix: Optional[str] = None,
) -> List[FinancialChunk]:
    """
    Financial Structural Enforcer & Dual Representation Builder:
    - Nếu nhóm chứa Bảng:
        - content_retrieval: Linearized Tuples (TableLinearizer) + Pre-table Narrative + Breadcrumbs + Footnotes.
        - content_generation: Ma trận bảng 2D Markdown + Pre-table Narrative + Footnotes + Post-table Notes + Breadcrumbs.
    - Nếu nhóm chỉ chứa Văn bản (Narrative):
        - content_retrieval: Narrative text + Breadcrumbs.
        - content_generation: Narrative text + Breadcrumbs.
    - Ràng buộc vỡ bảng: 0.0% (Bảng lớn > 1500 tokens được sub-group phân cấp theo hàng cha-con).
    """
    chunks_out: List[FinancialChunk] = []

    # Phân loại thành phần trong nhóm
    headings = [u for u in group_units if u.unit_type == "H"]
    tables = [u for u in group_units if u.unit_type == "T"]

    # Tiêu đề ngữ cảnh (Heading Context / Breadcrumb Stack)
    first_u = group_units[0]
    if getattr(first_u, "heading_path", None) and first_u.heading_path:
        main_heading = " > ".join(first_u.heading_path)
    elif headings and getattr(headings[-1], "heading_path", None) and headings[-1].heading_path:
        main_heading = " > ".join(headings[-1].heading_path)
    elif headings:
        main_heading = headings[0].text
    else:
        main_heading = first_u.heading_context or f"{ticker} FY{fiscal_year} 10-K Section"

    if tables:
        # Trường hợp 1: Nhóm có bảng số liệu
        for t_idx, t_unit in enumerate(tables, 1):
            table = t_unit.table_obj
            raw_caption = (table.caption if table and table.caption else "").strip()
            breadcrumb_str = " > ".join(t_unit.heading_path) if getattr(t_unit, "heading_path", None) else ""

            GENERIC_CAPTIONS = {"", "General Financial Information", "Table", "table", "Financial Information"}
            if raw_caption and raw_caption not in GENERIC_CAPTIONS:
                if breadcrumb_str:
                    caption = breadcrumb_str if raw_caption in breadcrumb_str else f"{breadcrumb_str} — {raw_caption}"
                else:
                    caption = raw_caption
            else:
                caption = breadcrumb_str or main_heading
            stmt_type = detect_statement_type(caption, str(table.rows[:2]) if table else "")
            unit_meta = parse_unit_metadata(table.unit if table else None)
            iso_dates = extract_iso_dates_from_period(
                table.period if table else None,
                ticker=ticker,
                fiscal_date_suffix=fiscal_date_suffix,
            )

            context_header = (
                f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | {caption}]\n"
                f"[Statement Type: {stmt_type}] | [Unit: {unit_meta['currency']} ({unit_meta['scale']})] | "
                f"[Period Dates: {', '.join(iso_dates) if iso_dates else 'N/A'}]"
            )

            # Xác định vị trí của bảng trong chuỗi các unit để phân tách chính xác:
            t_pos = group_units.index(t_unit) if t_unit in group_units else -1

            # Ranh giới cục bộ: chỉ lấy các đoạn văn thuộc phạm vi của bảng này
            prev_table_pos = -1
            for i in range(t_pos - 1, -1, -1):
                if group_units[i].unit_type == "T":
                    prev_table_pos = i
                    break

            next_table_pos = len(group_units)
            for i in range(t_pos + 1, len(group_units)):
                if group_units[i].unit_type == "T":
                    next_table_pos = i
                    break

            # Pre-paras: Chỉ nằm giữa bảng trước và bảng hiện tại
            pre_paras = [
                u for i, u in enumerate(group_units) 
                if (prev_table_pos < i < t_pos) and u.unit_type == "P"
            ]
            if len(pre_paras) > 2:
                pre_paras = pre_paras[-2:]

            # Post-footnotes: Nằm ngay sau bảng hiện tại và trước bảng kế tiếp
            post_fns = [
                u for i, u in enumerate(group_units) 
                if (t_pos < i < next_table_pos) and u.unit_type == "FN"
            ]

            # Post-paras: Nằm sau bảng hiện tại và trước bảng kế tiếp
            raw_post_paras = [
                u for i, u in enumerate(group_units) 
                if (t_pos < i < next_table_pos) and u.unit_type == "P"
            ]

            post_paras = []
            has_next_table = next_table_pos < len(group_units)
            for p_idx, p_u in enumerate(raw_post_paras):
                p_text_clean = p_u.text.strip()
                is_intro_sentence = (
                    p_text_clean.endswith(":") or
                    bool(re.search(r"(?:follow|follows|summarized as follows|as follows)\s*:\s*$", p_text_clean, re.I))
                )
                if is_intro_sentence and has_next_table:
                    continue
                if has_next_table and p_idx == len(raw_post_paras) - 1:
                    continue
                post_paras.append(p_u)

            # 1. Văn bản dẫn nhập (Contextual Narrative Intro)
            intro_text = ""
            if pre_paras:
                intro_body = "\n\n".join(p.text for p in pre_paras).strip()
                if intro_body:
                    intro_text = f"Contextual Narrative:\n{intro_body}\n\n"

            # 2. Footnotes gắn trực tiếp
            fn_texts = [fn.text for fn in post_fns]
            if table and table.footnotes:
                for tf in table.footnotes:
                    if tf not in fn_texts:
                        fn_texts.append(tf)
            footnotes_block = f"\n\nFootnotes:\n" + "\n".join(fn_texts) if fn_texts else ""

            # 3. Post-table commentary (nếu có, đặt sau bảng, không đặt trước bảng)
            post_text = ""
            if post_paras:
                post_body = "\n\n".join(p.text for p in post_paras).strip()
                if post_body:
                    post_text = f"\n\nPost-Table Notes:\n{post_body}"

            included_units = [t_unit] + pre_paras + post_fns + post_paras
            if headings:
                included_units.extend(headings)
            source_pages = sorted(set(
                p for u in included_units for p in (
                    u.table_obj.source_pages if (getattr(u, "table_obj", None) and u.table_obj.source_pages) else [u.page]
                )
            ))
            qc_warning = any(u.qc_status == "warning" for u in included_units)

            table_page = (
                t_unit.table_obj.source_pages[0]
                if (getattr(t_unit, "table_obj", None) and t_unit.table_obj.source_pages)
                else t_unit.page
            )

            headers = table.headers if table else []
            rows = table.rows if table else []
            full_md_table = render_markdown_table(headers, rows)
            full_gen_content = f"{context_header}\n\n{intro_text}{full_md_table}{footnotes_block}{post_text}"
            gen_tokens = len(enc.encode(full_gen_content))

            if gen_tokens <= MAX_TABLE_TOKENS:
                # 1.1 TABLE_ATOMIC Chunk
                linearized_body = linearize_financial_table(
                    headers=headers,
                    rows=rows,
                    caption=caption,
                    unit_meta=unit_meta,
                    period_dates=iso_dates,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                )
                retrieval_content = f"{context_header}\n\n{intro_text}Financial Fact Tuples:\n{linearized_body}{footnotes_block}{post_text}"

                chunk_id = f"{ticker.lower()}_{fiscal_year}_m5_tbl_{chunk_index:04d}_{t_idx:02d}"
                c = FinancialChunk(
                    chunk_id=chunk_id,
                    document_id=dir_name,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
                    chunk_type=ChunkType.TABLE_ATOMIC.value,
                    chunk_level="child",
                    content=full_gen_content,
                    content_retrieval=retrieval_content,
                    content_generation=full_gen_content,
                    token_count=gen_tokens,
                    source_pages=source_pages,
                    has_table_fragmentation=False,
                    metadata={
                        "statement_type": stmt_type,
                        "logical_table_id": table.logical_table_id if table else f"tbl_{t_idx}",
                        "section_title": caption,
                        "section_hierarchy": list(getattr(t_unit, "heading_path", [])) if getattr(t_unit, "heading_path", None) else [main_heading],
                        "is_multi_page": table.is_multi_page if table else False,
                        "unit": unit_meta,
                        "period_dates": iso_dates,
                        "has_linearized_tuples": True,
                        "has_qc_warning": qc_warning,
                        "source_verification_badge": (
                            f"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {table_page} "
                            "có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                            if qc_warning else None
                        ),
                        "pdf_inspector": {
                            "pdf_page": table_page,
                            "highlight_target": "table_bbox",
                        },
                    },
                )
                chunks_out.append(c)

            else:
                # 1.2 Sub-group Split (> 1500 tokens): Cắt theo group header row
                sub_groups: List[List[List[str]]] = []
                curr_group: List[List[str]] = []

                for row in rows:
                    if is_group_header_row(row) and curr_group:
                        sub_groups.append(curr_group)
                        curr_group = [row]
                    else:
                        curr_group.append(row)
                        if len(curr_group) >= 25:
                            sub_groups.append(curr_group)
                            curr_group = []

                if curr_group:
                    sub_groups.append(curr_group)

                for g_idx, sub_rows in enumerate(sub_groups, 1):
                    sub_md = render_markdown_table(headers, sub_rows)
                    part_label = f" [Part {g_idx}/{len(sub_groups)}]"
                    sub_gen_content = f"{context_header}{part_label}\n\n{intro_text}{sub_md}{footnotes_block}{post_text}"
                    sub_tokens = len(enc.encode(sub_gen_content))

                    sub_linearized = linearize_financial_table(
                        headers=headers,
                        rows=sub_rows,
                        caption=f"{caption}{part_label}",
                        unit_meta=unit_meta,
                        period_dates=iso_dates,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                    )
                    sub_retrieval_content = f"{context_header}{part_label}\n\n{intro_text}Financial Fact Tuples:\n{sub_linearized}{footnotes_block}{post_text}"

                    chunk_id = f"{ticker.lower()}_{fiscal_year}_m5_tbl_{chunk_index:04d}_sub{g_idx:02d}"
                    c = FinancialChunk(
                        chunk_id=chunk_id,
                        document_id=dir_name,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        chunk_method=ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
                        chunk_type=ChunkType.TABLE_SUBGROUP.value,
                        chunk_level="child",
                        content=sub_gen_content,
                        content_retrieval=sub_retrieval_content,
                        content_generation=sub_gen_content,
                        token_count=sub_tokens,
                        source_pages=source_pages,
                        has_table_fragmentation=False,
                        metadata={
                            "statement_type": stmt_type,
                            "logical_table_id": table.logical_table_id if table else f"tbl_{t_idx}",
                            "section_title": caption,
                            "section_hierarchy": list(getattr(t_unit, "heading_path", [])) if getattr(t_unit, "heading_path", None) else [main_heading],
                            "subgroup_index": g_idx,
                            "total_subgroups": len(sub_groups),
                            "unit": unit_meta,
                            "period_dates": iso_dates,
                            "has_linearized_tuples": True,
                            "has_qc_warning": qc_warning,
                            "source_verification_badge": (
                                f"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {table_page} "
                                "có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                                if qc_warning else None
                            ),
                            "pdf_inspector": {
                                "pdf_page": table_page,
                                "highlight_target": "table_bbox",
                            },
                        },
                    )
                    chunks_out.append(c)

    else:
        # Trường hợp 2: Nhóm thuần Narrative (Văn bản thuyết minh, thảo luận)
        if all(u.unit_type == "H" for u in group_units):
            return []

        sec_header = f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | Section: {main_heading}]"
        narrative_parts = []
        
        top_title = main_heading.split(" > ")[-1]
        has_internal_headings = any(u.unit_type == "H" for u in group_units)

        if not has_internal_headings:
            narrative_parts.append(f"### {top_title}")

        for u in group_units:
            if u.unit_type == "H":
                h_text = u.text.strip()
                if h_text and not is_noise_or_fragment(h_text):
                    level = min(len(getattr(u, "heading_path", [])) + 1, 4)
                    prefix = "#" * max(level, 2)
                    narrative_parts.append(f"{prefix} {h_text}")
            elif u.unit_type in ("P", "FN"):
                narrative_parts.append(u.text)

        full_narrative = "\n\n".join(narrative_parts).strip()
        full_text = f"{sec_header}\n\n{full_narrative}"
        n_tokens = len(enc.encode(full_text))

        source_pages = sorted(set(u.page for u in group_units))
        qc_warning = any(u.qc_status == "warning" for u in group_units)

        first_u = group_units[0]
        sec_hierarchy = list(getattr(first_u, "heading_path", [])) if getattr(first_u, "heading_path", None) else [main_heading]

        if n_tokens <= NARRATIVE_SPLIT_TOKENS:
            chunk_id = f"{ticker.lower()}_{fiscal_year}_m5_narr_{chunk_index:04d}"
            c = _make_narrative_chunk(
                chunk_id=chunk_id,
                document_id=dir_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                part_text=full_text,
                part_tokens=n_tokens,
                main_heading=main_heading,
                sec_hierarchy=sec_hierarchy,
                source_pages=source_pages,
                qc_warning=qc_warning,
            )
            chunks_out.append(c)
        else:
            sentences = split_sentences(full_narrative)
            curr_sentences: List[str] = []
            curr_tokens = 0
            sub_idx = 0

            for sent in sentences:
                s_tokens = len(enc.encode(sent))
                if curr_tokens + s_tokens > NARRATIVE_SPLIT_TOKENS and curr_sentences:
                    sub_idx += 1
                    part_text = f"{sec_header} [Part {sub_idx}]\n\n" + " ".join(curr_sentences)
                    part_tokens = len(enc.encode(part_text))
                    cid = f"{ticker.lower()}_{fiscal_year}_m5_narr_{chunk_index:04d}_p{sub_idx:02d}"
                    chunks_out.append(
                        _make_narrative_chunk(
                            chunk_id=cid,
                            document_id=dir_name,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                            part_text=part_text,
                            part_tokens=part_tokens,
                            main_heading=main_heading,
                            sec_hierarchy=sec_hierarchy,
                            source_pages=source_pages,
                            qc_warning=qc_warning,
                            sub_part=sub_idx,
                        )
                    )
                    overlap_sentences: List[str] = []
                    overlap_toks = 0
                    for s in reversed(curr_sentences):
                        st = len(enc.encode(s))
                        if overlap_toks + st <= NARRATIVE_OVERLAP_TOKENS:
                            overlap_sentences.insert(0, s)
                            overlap_toks += st
                        else:
                            if not overlap_sentences:
                                overlap_sentences.append(s)
                            break
                    curr_sentences = list(overlap_sentences)
                    curr_tokens = sum(len(enc.encode(s)) for s in curr_sentences)

                curr_sentences.append(sent)
                curr_tokens += s_tokens

            if curr_sentences:
                part_tokens_preview = len(enc.encode(" ".join(curr_sentences)))
                if part_tokens_preview < 80 and chunks_out:
                    prev = chunks_out[-1]
                    if prev.chunk_type == ChunkType.NARRATIVE_SECTION.value and prev.metadata.get("section_title") == main_heading:
                        prev.content = prev.content + "\n\n" + " ".join(curr_sentences)
                        prev.content_retrieval = prev.content
                        prev.content_generation = prev.content
                        prev.token_count = len(enc.encode(prev.content))
                        curr_sentences = []
                if curr_sentences:
                    sub_idx += 1
                    part_text = f"{sec_header} [Part {sub_idx}]\n\n" + " ".join(curr_sentences)
                    part_tokens = len(enc.encode(part_text))
                    cid = f"{ticker.lower()}_{fiscal_year}_m5_narr_{chunk_index:04d}_p{sub_idx:02d}"
                    chunks_out.append(
                        _make_narrative_chunk(
                            chunk_id=cid,
                            document_id=dir_name,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                            part_text=part_text,
                            part_tokens=part_tokens,
                            main_heading=main_heading,
                            sec_hierarchy=sec_hierarchy,
                            source_pages=source_pages,
                            qc_warning=qc_warning,
                            sub_part=sub_idx,
                        )
                    )

    return chunks_out
