"""
method3_llm_assisted.py

Phương pháp 3 (Proposed Method): Comprehensive LLM-Assisted Structure-Aware Chunking
- Canonical JSON Structural Units ([H], [P], [T], [FN]).
- SectionStitcher tích hợp để nhận diện bảng vắt trang (Multi-page Tables).
- Batch LLM Semantic Grouping (DeepSeek-Chat):
    - Gửi danh sách 15-20 units cho LLM để xác định quan hệ ngữ nghĩa (đoạn văn giới thiệu + bảng số liệu + footnote).
    - LLM chỉ trả về danh sách ID nhóm, tuyệt đối không sửa hay sinh lại text.
    - Fallback Heuristic bằng code nếu API gặp sự cố.
- Financial Structural Enforcer (Hard Code Constraints):
    - Đảm bảo Table-Atomic (không cắt ngang dòng bảng).
    - Sub-group Split nếu bảng > 1500 tokens (cắt tại Group Header row, lặp lại Table Header + Đơn vị tính).
    - Gắn kết Footnotes trực tiếp vào bảng cha.
- Cơ chế Biểu diễn kép (Dual Representation):
    - content_retrieval: Breadcrumb + Narrative + Linearized Tuples (Semantic Tuples từ TableLinearizer) tối ưu cho BM25 và Dense Vector.
    - content_generation: Breadcrumb + Narrative + Ma trận bảng 2D Markdown + Footnotes sạch cho LLM suy luận.
    - content: Bản hiển thị chuẩn (content_generation).
- 1-Click PDF Inspector & Cảnh báo nguồn (Tuyệt đối KHÔNG có phần trăm độ tin cậy).
- Document Summary & Cross-Reference Index chunks.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import tiktoken
from dotenv import load_dotenv

from script.financial_chunker.enricher import (
    detect_statement_type,
    extract_iso_dates_from_period,
    parse_unit_metadata,
)
from script.financial_chunker.linearizer import linearize_financial_table
from script.financial_chunker.models import (
    ChunkMethod,
    ChunkType,
    FinancialChunk,
    StitchedDocument,
    StitchedTable,
)
from script.financial_chunker.stitcher import SectionStitcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

COMPANY_SPECS = [
    {"dir_name": "nvidia_2025_10k", "ticker": "NVDA", "fiscal_year": 2025},
    {"dir_name": "amd_10k_2025", "ticker": "AMD", "fiscal_year": 2025},
    {"dir_name": "apple_2025_10k", "ticker": "AAPL", "fiscal_year": 2025},
    {"dir_name": "intel_2025_10k", "ticker": "INTC", "fiscal_year": 2025},
]

OUTPUT_PARSING_ROOT = Path("output_parsing")
OUTPUT_CHUNKING_ROOT = Path("output_chunking")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = "deepseek-chat"

ENCODING_NAME = "cl100k_base"
BATCH_UNIT_SIZE = 16
MAX_TABLE_TOKENS = 1500
TARGET_GROUP_MAX_TOKENS = 1100
NARRATIVE_SPLIT_TOKENS = 750
NARRATIVE_OVERLAP_TOKENS = 90

FOOTNOTE_PATTERN = re.compile(
    r"^\s*(?:\([0-9a-zA-Z*†‡§#]\)|\[[0-9a-zA-Z*†‡§#]\]|\*+|[0-9]{1,2}\.?\s+[A-Z])"
)


@dataclass
class StructuralUnit:
    unit_id: str
    unit_type: str  # "H", "P", "T", "FN"
    page: int
    text: str
    tokens: int
    preview: str
    heading_context: str = ""
    table_obj: Optional[StitchedTable] = None
    qc_status: str = "pass"


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not headers and not rows:
        return ""

    num_cols = len(headers) if headers else (len(rows[0]) if rows else 1)
    clean_headers = [h.replace("\n", " ").strip() for h in headers]
    while len(clean_headers) < num_cols:
        clean_headers.append(f"Col {len(clean_headers)+1}")

    header_line = "| " + " | ".join(clean_headers) + " |"
    separator_line = "| " + " | ".join(["---"] * num_cols) + " |"

    rendered_rows = []
    for row in rows:
        clean_cells = [str(c).replace("\n", " ").strip() for c in row]
        while len(clean_cells) < num_cols:
            clean_cells.append("")
        rendered_rows.append("| " + " | ".join(clean_cells[:num_cols]) + " |")

    return "\n".join([header_line, separator_line] + rendered_rows)


def is_group_header_row(row: List[str]) -> bool:
    if not row:
        return False
    first_cell = row[0].strip()
    if not first_cell:
        return False
    rest_cells = [c.strip() for c in row[1:]]
    return all(c == "" or c == "-" or c == "—" for c in rest_cells)


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<!\d\.)(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def is_table_separator(line: str) -> bool:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return all(re.match(r"^:?-+:?$", c) for c in cells if c)


def detect_table_fragmentation(chunk_text: str) -> Tuple[bool, List[str]]:
    lines = chunk_text.splitlines()
    reasons: List[str] = []
    in_table = False
    table_has_header = False

    for i, line in enumerate(lines):
        if is_table_row(line):
            if not in_table:
                in_table = True
                if i + 1 < len(lines) and is_table_separator(lines[i + 1]):
                    table_has_header = True
                else:
                    table_has_header = False
                    reasons.append(f"starts_mid_table_at_line_{i}")
        else:
            if in_table:
                if not table_has_header:
                    reasons.append(f"table_lacks_header_before_line_{i}")
                in_table = False
                table_has_header = False

    return len(reasons) > 0, reasons


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
    unit_counter = 0
    current_heading = "General Financial Information"

    for p_data in sorted_pages:
        page_obj = p_data.get("page", {})
        pdf_page = page_obj.get("pdf_page", 1)
        page_qc = page_obj.get("qc", {}).get("status", "pass")
        blocks = page_obj.get("blocks", [])

        last_table_unit_id: Optional[str] = None

        for b_idx, block in enumerate(blocks):
            b_type = block.get("block_type")
            b_id = block.get("block_id", f"p{pdf_page}_b{b_idx}")

            if b_type == "heading":
                h_text = block.get("text", "").strip()
                if not h_text:
                    continue
                current_heading = h_text
                unit_counter += 1
                tokens = len(enc.encode(h_text))
                units.append(
                    StructuralUnit(
                        unit_id=f"u{unit_counter:04d}",
                        unit_type="H",
                        page=pdf_page,
                        text=h_text,
                        tokens=tokens,
                        preview=f"Heading: {h_text[:70]}",
                        heading_context=current_heading,
                        qc_status=page_qc,
                    )
                )

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

                units.append(
                    StructuralUnit(
                        unit_id=u_id,
                        unit_type="T",
                        page=pdf_page,
                        text=md_repr,
                        tokens=t_tokens,
                        preview=f"Table: {caption[:60]} ({len(table_to_use.rows)} rows)",
                        heading_context=current_heading,
                        table_obj=table_to_use,
                        qc_status=table_to_use.qc_status,
                    )
                )

            elif b_type == "paragraph":
                p_text = block.get("text", "").strip()
                if not p_text:
                    continue

                p_tokens = len(enc.encode(p_text))

                # Kiểm tra xem có phải footnote của bảng liền trước không
                is_fn = bool(FOOTNOTE_PATTERN.match(p_text))
                u_type = "FN" if (is_fn and last_table_unit_id) else "P"

                unit_counter += 1
                units.append(
                    StructuralUnit(
                        unit_id=f"u{unit_counter:04d}",
                        unit_type=u_type,
                        page=pdf_page,
                        text=p_text,
                        tokens=p_tokens,
                        preview=f"{u_type}: {p_text[:75]}",
                        heading_context=current_heading,
                        qc_status=page_qc,
                    )
                )

    return units


def request_llm_semantic_grouping(
    batch_units: List[StructuralUnit],
    ticker: str,
    fiscal_year: int,
) -> List[List[str]]:
    """
    Gửi batch 15-20 Structural Units cho DeepSeek-Chat để xác định ranh giới gom cụm ngữ nghĩa.
    LLM chỉ trả về danh sách ID, không sinh hay sửa nội dung.
    """
    if not DEEPSEEK_API_KEY:
        return fallback_heuristic_grouping(batch_units)

    items_payload = [
        {
            "id": u.unit_id,
            "type": u.unit_type,
            "tokens": u.tokens,
            "preview": u.preview,
        }
        for u in batch_units
    ]

    system_prompt = (
        "You are an expert financial document structural chunking assistant. "
        "You are given an ordered sequence of structural units from a 10-K report: "
        "H (Heading), P (Paragraph), T (Table), FN (Footnote). "
        "Your task: Group consecutive units into semantically cohesive chunks. "
        "Rules:\n"
        "1. Target chunk size: 400 - 900 tokens. Never exceed 1100 tokens total in a group.\n"
        "2. Keep a Table (T) together with its introductory paragraph (P) and any attached Footnotes (FN) if total tokens <= 1100.\n"
        "3. A large Table (tokens > 900) MUST form its own single-element group.\n"
        "4. Keep a Heading (H) with its following content (never leave H alone if following content fits).\n"
        "5. ONLY group adjacent units in exact sequence. Do not skip or reorder.\n"
        "6. Do NOT group across major Note or Item boundaries (e.g. Note 1 and Note 2 must be in different groups).\n"
        "7. Output strictly valid JSON with a single key 'groups', e.g.:\n"
        '{"groups": [["u0001", "u0002", "u0003"], ["u0004", "u0005"]]}\n'
        "Every input unit ID must appear in exactly one group."
    )

    user_prompt = (
        f"Company: {ticker} FY{fiscal_year} 10-K\n"
        f"Structural units to group:\n{json.dumps(items_payload, ensure_ascii=False, indent=2)}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    endpoint = f"{DEEPSEEK_BASE_URL}/chat/completions"

    for attempt in range(1, 3):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                clean_json = re.sub(r"^```(?:json)?\s*", "", content)
                clean_json = re.sub(r"\s*```$", "", clean_json).strip()
                parsed = json.loads(clean_json)
                groups = parsed.get("groups", [])
                if isinstance(groups, list) and groups:
                    all_returned_ids = [item for g in groups for item in g]
                    input_ids = [u.unit_id for u in batch_units]
                    if set(all_returned_ids) == set(input_ids):
                        return groups
        except Exception:
            pass
        time.sleep(1)

    return fallback_heuristic_grouping(batch_units)


def fallback_heuristic_grouping(batch_units: List[StructuralUnit]) -> List[List[str]]:
    """
    Fallback Heuristic bằng code nếu LLM API gặp lỗi hoặc trả về không hợp lệ.
    """
    groups: List[List[str]] = []
    curr_group: List[str] = []
    curr_tokens = 0
    has_table = False

    for u in batch_units:
        # Nếu gặp bảng lớn đơn lẻ
        if u.unit_type == "T" and u.tokens > 800:
            if curr_group:
                groups.append(curr_group)
                curr_group = []
                curr_tokens = 0
                has_table = False
            groups.append([u.unit_id])
            continue

        # Nếu thêm vào sẽ vượt ngưỡng
        if curr_tokens + u.tokens > TARGET_GROUP_MAX_TOKENS and curr_group:
            groups.append(curr_group)
            curr_group = [u.unit_id]
            curr_tokens = u.tokens
            has_table = (u.unit_type == "T")
        else:
            curr_group.append(u.unit_id)
            curr_tokens += u.tokens
            if u.unit_type == "T":
                has_table = True

    if curr_group:
        groups.append(curr_group)

    return groups


def build_dual_representation_chunk(
    group_units: List[StructuralUnit],
    ticker: str,
    fiscal_year: int,
    dir_name: str,
    chunk_index: int,
    enc: tiktoken.Encoding,
) -> List[FinancialChunk]:
    """
    Financial Structural Enforcer & Dual Representation Builder:
    - Nếu nhóm chứa Bảng:
        - content_retrieval: Linearized Tuples (TableLinearizer) + Narrative + Breadcrumbs.
        - content_generation: Ma trận bảng 2D Markdown + Narrative + Footnotes + Breadcrumbs.
    - Nếu nhóm chỉ chứa Văn bản (Narrative):
        - content_retrieval: Narrative text + Breadcrumbs.
        - content_generation: Narrative text + Breadcrumbs.
    - Ràng buộc vỡ bảng: 0.0% (Bảng lớn > 1500 tokens được sub-group phân cấp theo hàng cha-con).
    """
    chunks_out: List[FinancialChunk] = []

    # Phân loại thành phần trong nhóm
    headings = [u for u in group_units if u.unit_type == "H"]
    paragraphs = [u for u in group_units if u.unit_type == "P"]
    tables = [u for u in group_units if u.unit_type == "T"]
    footnotes = [u for u in group_units if u.unit_type == "FN"]

    source_pages = sorted(list({u.page for u in group_units}))
    qc_warning = any(u.qc_status == "warning" for u in group_units)

    # Tiêu đề ngữ cảnh (Heading Context)
    main_heading = headings[0].text if headings else group_units[0].heading_context or f"{ticker} FY{fiscal_year} 10-K Section"

    if tables:
        # Trường hợp 1: Nhóm có bảng số liệu
        for t_idx, t_unit in enumerate(tables, 1):
            table = t_unit.table_obj
            caption = table.caption if table and table.caption else main_heading
            stmt_type = detect_statement_type(caption, str(table.rows[:2]) if table else "")
            unit_meta = parse_unit_metadata(table.unit if table else None)
            iso_dates = extract_iso_dates_from_period(table.period if table else None)

            context_header = (
                f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | {caption}]\n"
                f"[Statement Type: {stmt_type}] | [Unit: {unit_meta['currency']} ({unit_meta['scale']})] | "
                f"[Period Dates: {', '.join(iso_dates) if iso_dates else 'N/A'}]"
            )

            # Văn bản dẫn nhập (Narrative Intro)
            intro_text = "\n\n".join(p.text for p in paragraphs)
            if intro_text:
                intro_text = f"Contextual Narrative:\n{intro_text}\n\n"

            # Footnotes
            fn_texts = [fn.text for fn in footnotes]
            if table and table.footnotes:
                for tf in table.footnotes:
                    if tf not in fn_texts:
                        fn_texts.append(tf)
            footnotes_block = f"\n\nFootnotes:\n" + "\n".join(fn_texts) if fn_texts else ""

            # Kiểm tra kích thước bảng để quyết định Table-Atomic hay Sub-group Split
            headers = table.headers if table else []
            rows = table.rows if table else []
            full_md_table = render_markdown_table(headers, rows)
            full_gen_content = f"{context_header}\n\n{intro_text}{full_md_table}{footnotes_block}"
            gen_tokens = len(enc.encode(full_gen_content))

            if gen_tokens <= MAX_TABLE_TOKENS:
                # 1.1 TABLE_ATOMIC Chunk
                # Xây dựng content_retrieval qua TableLinearizer
                linearized_body = linearize_financial_table(
                    headers=headers,
                    rows=rows,
                    caption=caption,
                    unit_meta=unit_meta,
                    period_dates=iso_dates,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                )
                retrieval_content = f"{context_header}\n\n{intro_text}Financial Fact Tuples:\n{linearized_body}{footnotes_block}"

                chunk_id = f"{ticker.lower()}_{fiscal_year}_m3_tbl_{chunk_index:04d}_{t_idx:02d}"
                c = FinancialChunk(
                    chunk_id=chunk_id,
                    document_id=dir_name,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                    chunk_type=ChunkType.TABLE_ATOMIC.value,
                    content=full_gen_content,
                    content_retrieval=retrieval_content,
                    content_generation=full_gen_content,
                    token_count=gen_tokens,
                    source_pages=source_pages,
                    has_table_fragmentation=False,
                    metadata={
                        "statement_type": stmt_type,
                        "logical_table_id": table.logical_table_id if table else f"tbl_{t_idx}",
                        "is_multi_page": table.is_multi_page if table else False,
                        "unit": unit_meta,
                        "period_dates": iso_dates,
                        "has_linearized_tuples": True,
                        "has_qc_warning": qc_warning,
                        "source_verification_badge": (
                            f"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {source_pages[0]} "
                            "có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                            if qc_warning else None
                        ),
                        "pdf_inspector": {
                            "pdf_page": source_pages[0],
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
                    sub_gen_content = f"{context_header}{part_label}\n\n{intro_text}{sub_md}{footnotes_block}"
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
                    sub_retrieval_content = f"{context_header}{part_label}\n\n{intro_text}Financial Fact Tuples:\n{sub_linearized}{footnotes_block}"

                    chunk_id = f"{ticker.lower()}_{fiscal_year}_m3_tbl_{chunk_index:04d}_sub{g_idx:02d}"
                    c = FinancialChunk(
                        chunk_id=chunk_id,
                        document_id=dir_name,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                        chunk_type=ChunkType.TABLE_SUBGROUP.value,
                        content=sub_gen_content,
                        content_retrieval=sub_retrieval_content,
                        content_generation=sub_gen_content,
                        token_count=sub_tokens,
                        source_pages=source_pages,
                        has_table_fragmentation=False,
                        metadata={
                            "statement_type": stmt_type,
                            "logical_table_id": table.logical_table_id if table else f"tbl_{t_idx}",
                            "subgroup_index": g_idx,
                            "total_subgroups": len(sub_groups),
                            "unit": unit_meta,
                            "period_dates": iso_dates,
                            "has_linearized_tuples": True,
                            "has_qc_warning": qc_warning,
                            "source_verification_badge": (
                                f"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {source_pages[0]} "
                                "có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                                if qc_warning else None
                            ),
                            "pdf_inspector": {
                                "pdf_page": source_pages[0],
                                "highlight_target": "table_bbox",
                            },
                        },
                    )
                    chunks_out.append(c)

    else:
        # Trường hợp 2: Nhóm thuần Narrative (Văn bản thuyết minh, thảo luận)
        sec_header = f"[Document: {ticker} Corporation FY{fiscal_year} 10-K | Section: {main_heading}]"
        narrative_parts = []
        for u in group_units:
            if u.unit_type == "H":
                narrative_parts.append(f"### {u.text}")
            elif u.unit_type in ("P", "FN"):
                narrative_parts.append(u.text)

        full_narrative = "\n\n".join(narrative_parts).strip()
        full_text = f"{sec_header}\n\n{full_narrative}"
        n_tokens = len(enc.encode(full_text))

        if n_tokens <= TARGET_GROUP_MAX_TOKENS:
            chunk_id = f"{ticker.lower()}_{fiscal_year}_m3_narr_{chunk_index:04d}"
            c = FinancialChunk(
                chunk_id=chunk_id,
                document_id=dir_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                chunk_type=ChunkType.NARRATIVE_SECTION.value,
                content=full_text,
                content_retrieval=full_text,
                content_generation=full_text,
                token_count=n_tokens,
                source_pages=source_pages,
                has_table_fragmentation=False,
                metadata={
                    "section_title": main_heading,
                    "has_linearized_tuples": False,
                    "has_qc_warning": qc_warning,
                    "source_verification_badge": (
                        f"⚠️ Lưu ý kiểm tra nguồn: Văn bản tại Trang {source_pages[0]} "
                        "có định dạng phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                        if qc_warning else None
                    ),
                    "pdf_inspector": {
                        "pdf_page": source_pages[0],
                        "highlight_target": "text_block",
                    },
                },
            )
            chunks_out.append(c)
        else:
            # Cắt trượt câu nếu quá dài
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
                    cid = f"{ticker.lower()}_{fiscal_year}_m3_narr_{chunk_index:04d}_p{sub_idx:02d}"
                    chunks_out.append(
                        FinancialChunk(
                            chunk_id=cid,
                            document_id=dir_name,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                            chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                            chunk_type=ChunkType.NARRATIVE_SECTION.value,
                            content=part_text,
                            content_retrieval=part_text,
                            content_generation=part_text,
                            token_count=part_tokens,
                            source_pages=source_pages,
                            has_table_fragmentation=False,
                            metadata={"section_title": main_heading, "sub_part": sub_idx},
                        )
                    )
                    # Overlap câu cuối
                    overlap_sentences = curr_sentences[-2:] if len(curr_sentences) >= 2 else curr_sentences
                    curr_sentences = list(overlap_sentences)
                    curr_tokens = sum(len(enc.encode(s)) for s in curr_sentences)

                curr_sentences.append(sent)
                curr_tokens += s_tokens

            if curr_sentences:
                sub_idx += 1
                part_text = f"{sec_header} [Part {sub_idx}]\n\n" + " ".join(curr_sentences)
                part_tokens = len(enc.encode(part_text))
                cid = f"{ticker.lower()}_{fiscal_year}_m3_narr_{chunk_index:04d}_p{sub_idx:02d}"
                chunks_out.append(
                    FinancialChunk(
                        chunk_id=cid,
                        document_id=dir_name,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                        chunk_type=ChunkType.NARRATIVE_SECTION.value,
                        content=part_text,
                        content_retrieval=part_text,
                        content_generation=part_text,
                        token_count=part_tokens,
                        source_pages=source_pages,
                        has_table_fragmentation=False,
                        metadata={"section_title": main_heading, "sub_part": sub_idx},
                    )
                )

    return chunks_out


def create_specialized_financial_chunks(
    pages_data: List[Dict[str, Any]],
    ticker: str,
    fiscal_year: int,
    dir_name: str,
    enc: tiktoken.Encoding,
) -> List[FinancialChunk]:
    """
    Tạo 2 chunk chuyên biệt: Document Summary và Cross-Reference Index.
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
                chunk_id=f"{ticker.lower()}_{fiscal_year}_m3_xref_index",
                document_id=dir_name,
                ticker=ticker,
                fiscal_year=fiscal_year,
                chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
                chunk_type=ChunkType.CROSS_REFERENCE_INDEX.value,
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
            chunk_id=f"{ticker.lower()}_{fiscal_year}_m3_doc_summary",
            document_id=dir_name,
            ticker=ticker,
            fiscal_year=fiscal_year,
            chunk_method=ChunkMethod.METHOD3_LLM_ASSISTED.value,
            chunk_type=ChunkType.DOCUMENT_SUMMARY.value,
            content=summary_text,
            content_retrieval=summary_text,
            content_generation=summary_text,
            token_count=s_tokens,
            source_pages=[1],
            has_table_fragmentation=False,
            metadata={"summary_scope": "executive_financial_overview"},
        )
    )

    return specialized_chunks


def run_method3_for_company(
    company_spec: Dict[str, Any],
) -> Dict[str, Any]:
    dir_name = company_spec["dir_name"]
    ticker = company_spec["ticker"]
    fiscal_year = company_spec["fiscal_year"]

    company_parsing_dir = OUTPUT_PARSING_ROOT / dir_name
    pages_dir = company_parsing_dir / "pages"

    if not pages_dir.exists():
        return {"company": dir_name, "status": "skipped_missing_pages", "chunk_count": 0}

    json_files = sorted(
        pages_dir.glob("page_*.json"),
        key=lambda p: int(re.search(r"page_(\d+)\.json", p.name).group(1)),
    )

    if not json_files:
        return {"company": dir_name, "status": "no_pages_found", "chunk_count": 0}

    pages_data = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            pages_data.append(json.load(f))

    enc = tiktoken.get_encoding(ENCODING_NAME)

    # 1. Ghép nối bảng đa trang với SectionStitcher
    stitcher = SectionStitcher(column_match_threshold=0.75)
    stitched_doc = stitcher.stitch_company_pages(
        pages_data=pages_data,
        document_id=dir_name,
        ticker=ticker,
        fiscal_year=fiscal_year,
    )
    multi_page_table_count = sum(1 for t in stitched_doc.tables if t.is_multi_page)

    # 2. Trích xuất Structural Units tuần tự
    units = extract_structural_units(pages_data, stitched_doc, enc)
    unit_map = {u.unit_id: u for u in units}

    # 3. Gom cụm theo batch gửi LLM qua ThreadPoolExecutor (max_workers=8)
    batches = [units[b_i : b_i + BATCH_UNIT_SIZE] for b_i in range(0, len(units), BATCH_UNIT_SIZE)]
    total_batches = len(batches)
    print(f"[{ticker}] Tổng số Structural Units: {len(units)} across {len(json_files)} pages. Batch size: {BATCH_UNIT_SIZE} (Total batches: {total_batches})", flush=True)

    def process_batch(item: Tuple[int, List[StructuralUnit]]) -> Tuple[int, List[List[str]]]:
        idx, b = item
        res = request_llm_semantic_grouping(b, ticker, fiscal_year)
        return idx, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batch_results = list(executor.map(process_batch, enumerate(batches)))

    batch_results.sort(key=lambda x: x[0])
    all_unit_groups: List[List[str]] = []
    for _, grps in batch_results:
        all_unit_groups.extend(grps)
    llm_api_calls = total_batches

    # 4. Financial Structural Enforcer & Dual Representation
    final_chunks: List[FinancialChunk] = []
    chunk_index = 0

    for group_ids in all_unit_groups:
        group_units = [unit_map[uid] for uid in group_ids if uid in unit_map]
        if not group_units:
            continue

        chunk_index += 1
        built_chunks = build_dual_representation_chunk(
            group_units=group_units,
            ticker=ticker,
            fiscal_year=fiscal_year,
            dir_name=dir_name,
            chunk_index=chunk_index,
            enc=enc,
        )
        final_chunks.extend(built_chunks)

    # 5. Thêm Specialized Chunks
    spec_chunks = create_specialized_financial_chunks(
        pages_data=pages_data,
        ticker=ticker,
        fiscal_year=fiscal_year,
        dir_name=dir_name,
        enc=enc,
    )
    final_chunks.extend(spec_chunks)

    # 6. Kiểm tra toàn diện chất lượng Chunks
    table_atomic_count = 0
    table_subgroup_count = 0
    narrative_count = 0
    specialized_count = 0
    fragmented_chunks = 0
    token_counts = []

    for c in final_chunks:
        token_counts.append(c.token_count)
        if c.chunk_type == ChunkType.TABLE_ATOMIC.value:
            table_atomic_count += 1
        elif c.chunk_type == ChunkType.TABLE_SUBGROUP.value:
            table_subgroup_count += 1
        elif c.chunk_type == ChunkType.NARRATIVE_SECTION.value:
            narrative_count += 1
        else:
            specialized_count += 1

        is_frag, _ = detect_table_fragmentation(c.content_generation)
        if is_frag:
            fragmented_chunks += 1
            c.has_table_fragmentation = True

    token_counts_sorted = sorted(token_counts)
    min_tokens = token_counts_sorted[0] if token_counts else 0
    max_tokens = token_counts_sorted[-1] if token_counts else 0
    mean_tokens = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0
    median_tokens = token_counts_sorted[len(token_counts_sorted) // 2] if token_counts else 0

    # 7. Lưu output vào folder riêng proposed_llm_assisted_structure
    comp_chunk_dir = OUTPUT_CHUNKING_ROOT / dir_name / "proposed_llm_assisted_structure"
    comp_chunk_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = comp_chunk_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in final_chunks:
            f.write(c.to_json() + "\n")

    summary_info = {
        "company": dir_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "method": ChunkMethod.METHOD3_LLM_ASSISTED.value,
        "total_structural_units": len(units),
        "total_chunks_created": len(final_chunks),
        "table_atomic_chunks": table_atomic_count,
        "table_subgroup_chunks": table_subgroup_count,
        "narrative_chunks": narrative_count,
        "specialized_chunks": specialized_count,
        "multi_page_tables_stitched": multi_page_table_count,
        "fragmented_table_chunks": fragmented_chunks,
        "fragmentation_rate": round(fragmented_chunks / len(final_chunks), 4) if final_chunks else 0.0,
        "llm_api_batches": llm_api_calls,
        "token_distribution": {
            "min": min_tokens,
            "max": max_tokens,
            "mean": mean_tokens,
            "median": median_tokens,
        },
        "output_file": str(jsonl_path),
    }

    summary_path = comp_chunk_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, ensure_ascii=False, indent=2)

    print(
        f"[{ticker}] DONE: {len(final_chunks)} chunks (Atomic Tbl: {table_atomic_count}, "
        f"Sub-group Tbl: {table_subgroup_count}, Narr: {narrative_count}, Spec: {specialized_count}) | "
        f"Frag: {fragmented_chunks} ({summary_info['fragmentation_rate']*100:.1f}%) | "
        f"Tokens mean: {mean_tokens}, median: {median_tokens}.",
        flush=True,
    )

    return summary_info


def main() -> None:
    start_time = time.time()
    print("=================================================================", flush=True)
    print("BẮT ĐẦU CHẠY PHƯƠNG PHÁP 3: PROPOSED LLM-ASSISTED STRUCTURE CHUNKING", flush=True)
    print("=================================================================", flush=True)

    results = []
    for spec in COMPANY_SPECS:
        print(f"\n>>> Đang xử lý: {spec['ticker']} ({spec['dir_name']}) ...", flush=True)
        res = run_method3_for_company(spec)
        results.append(res)

    total_time = round(time.time() - start_time, 2)
    print("\n=================================================================", flush=True)
    print(f"HOÀN TẤT PHƯƠNG PHÁP 3 CHO CẢ 4 TẬP ĐOÀN! Thời gian: {total_time}s", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
