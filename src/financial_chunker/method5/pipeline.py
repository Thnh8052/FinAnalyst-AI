"""
pipeline.py - Method 5 Orchestration, Batch Processing, and Output Export
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from typing import Any, Dict, List, Tuple

import tiktoken

from financial_chunker.models import ChunkMethod, ChunkType, FinancialChunk
from financial_chunker.stitcher import SectionStitcher
from financial_chunker.method5.common import (
    BATCH_UNIT_SIZE,
    COMPANY_SPECS,
    detect_table_fragmentation,
    ENCODING_NAME,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_PARSING_ROOT,
    StructuralUnit,
)
from financial_chunker.method5.consolidator import (
    consolidate_semantic_groups,
    extract_structural_units,
)
from financial_chunker.method5.child_builder import build_dual_representation_chunk
from financial_chunker.method5.grouper import (
    request_llm_semantic_grouping,
    split_into_safe_batches,
)
from financial_chunker.method5.parent_builder import (
    build_parent_chunks,
    create_specialized_financial_chunks,
)


def run_method5_for_company(
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
    batches = split_into_safe_batches(units, BATCH_UNIT_SIZE)
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

    # 3.5 Áp dụng Semantic & Layout Consolidation Engine:
    # - Triệt tiêu triệt để các nhóm tiêu đề mồ côi (Heading-Only chunks).
    # - Tách các bảng thành đơn vị độc lập (Atomic Table Splitting).
    # - Gắn câu kết luận / ghi chú sau bảng vào Table làm Post-Table Notes.
    # - Gom cụm các đoạn văn ngắn (< 120 tokens) thành chunk hoàn chỉnh, loại bỏ hoàn toàn chunk ngắn cụt.
    all_unit_groups = consolidate_semantic_groups(all_unit_groups, unit_map, enc)

    # 4. Financial Structural Enforcer & Dual Representation (Child Chunks)
    child_chunks: List[FinancialChunk] = []
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
            fiscal_date_suffix=stitched_doc.fiscal_date_suffix,
        )
        child_chunks.extend(built_chunks)

    # 5. Xây dựng Parent Chunks và liên kết Phân cấp Cha - Con (Parent-Child Hierarchy)
    parent_chunks, child_chunks = build_parent_chunks(
        units=units,
        child_chunks=child_chunks,
        ticker=ticker,
        fiscal_year=fiscal_year,
        dir_name=dir_name,
        enc=enc,
    )

    # 6. Thêm Specialized Chunks (Tier 0: doc_summary, xref_index, toc)
    spec_chunks = create_specialized_financial_chunks(
        pages_data=pages_data,
        ticker=ticker,
        fiscal_year=fiscal_year,
        dir_name=dir_name,
        enc=enc,
    )

    final_chunks: List[FinancialChunk] = spec_chunks + parent_chunks + child_chunks

    # 7. Kiểm tra toàn diện chất lượng Chunks
    table_atomic_count = 0
    table_subgroup_count = 0
    narrative_count = 0
    parent_count = 0
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
        elif c.chunk_type == ChunkType.PARENT_SECTION.value:
            parent_count += 1
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

    # 8. Lưu output vào folder riêng method5_proposed_golden_hybrid
    comp_chunk_dir = OUTPUT_CHUNKING_ROOT / dir_name / "method5_proposed_golden_hybrid"
    comp_chunk_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = comp_chunk_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in final_chunks:
            f.write(c.to_json() + "\n")

    parent_jsonl_path = comp_chunk_dir / "parent_chunks.jsonl"
    with open(parent_jsonl_path, "w", encoding="utf-8") as f:
        for c in parent_chunks:
            f.write(c.to_json() + "\n")

    summary_info = {
        "company": dir_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "method": ChunkMethod.METHOD5_GOLDEN_HYBRID.value,
        "total_structural_units": len(units),
        "total_chunks_created": len(final_chunks),
        "parent_chunks": parent_count,
        "child_chunks": len(child_chunks),
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
        "parent_output_file": str(parent_jsonl_path),
    }

    summary_path = comp_chunk_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, ensure_ascii=False, indent=2)

    print(
        f"[{ticker}] DONE: {len(final_chunks)} chunks (Parent: {parent_count}, Child: {len(child_chunks)} "
        f"[Atomic Tbl: {table_atomic_count}, Sub-group Tbl: {table_subgroup_count}, Narr: {narrative_count}], "
        f"Spec: {specialized_count}) | Frag: {fragmented_chunks} ({summary_info['fragmentation_rate']*100:.1f}%) | "
        f"Tokens mean: {mean_tokens}, median: {median_tokens}.",
        flush=True,
    )

    return summary_info


def main() -> None:
    start_time = time.time()
    print("=================================================================", flush=True)
    print("BẮT ĐẦU CHẠY PHƯƠNG PHÁP 5: PROPOSED GOLDEN HYBRID CHUNKING", flush=True)
    print("=================================================================", flush=True)

    results = []
    for spec in COMPANY_SPECS:
        print(f"\n>>> Đang xử lý: {spec['ticker']} ({spec['dir_name']}) ...", flush=True)
        res = run_method5_for_company(spec)
        results.append(res)

    total_time = round(time.time() - start_time, 2)
    print("\n=================================================================", flush=True)
    print(f"HOÀN TẤT PHƯƠNG PHÁP 5 CHO TẤT CẢ {len(COMPANY_SPECS)} HỒ SƠ DOANH NGHIỆP! Thời gian: {total_time}s", flush=True)
    print("=================================================================", flush=True)
