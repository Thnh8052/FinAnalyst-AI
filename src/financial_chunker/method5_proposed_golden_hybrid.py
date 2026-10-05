"""
method5_proposed_golden_hybrid.py

Phương pháp 5 (Proposed Golden Hybrid): Comprehensive LLM-Assisted Structure-Aware Chunking.

FACADE MODULE:
File này đóng vai trò Facade Entrypoint để bảo đảm 100% tính tương thích ngược (Backward Compatibility)
cho toàn bộ codebase, tests, scripts và CLI runner.

Mã nguồn chi tiết đã được module hóa chuyên sâu theo nguyên tắc Clean Architecture tại:
    src/financial_chunker/method5/
        ├── common.py           # Constants, StructuralUnit, Table & Text utilities
        ├── headings.py         # Heading hierarchy, breadcrumbs, noise filtering
        ├── grouper.py          # LLM API semantic grouping, safe batching, heuristic fallback
        ├── consolidator.py     # Structural unit extraction, semantic group consolidation
        ├── child_builder.py    # Dual representation child chunk builders (Tuples + Markdown)
        ├── parent_builder.py   # Multi-tier parent chunks & document-level specialized chunks
        ├── pipeline.py         # Company runner, batch threading, JSONL export
        └── __init__.py         # Package interface
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-export toàn bộ API từ subpackage method5
from financial_chunker.method5 import (
    BATCH_UNIT_SIZE,
    COMPANY_SPECS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    ENCODING_NAME,
    FOOTNOTE_PATTERN,
    MAX_PARENT_TOKENS,
    MAX_TABLE_TOKENS,
    NARRATIVE_OVERLAP_TOKENS,
    NARRATIVE_SPLIT_TOKENS,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_PARSING_ROOT,
    StructuralUnit,
    TARGET_GROUP_MAX_TOKENS,
    _make_narrative_chunk,
    build_dual_representation_chunk,
    build_parent_chunks,
    consolidate_semantic_groups,
    create_specialized_financial_chunks,
    detect_table_fragmentation,
    extract_structural_units,
    fallback_heuristic_grouping,
    is_major_subsection,
    is_noise_or_fragment,
    is_table_row,
    is_table_separator,
    is_top_level_heading,
    main,
    render_markdown_table,
    request_llm_semantic_grouping,
    run_method5_for_company,
    split_into_safe_batches,
    split_sentences,
    update_heading_stack,
)

__all__ = [
    "StructuralUnit",
    "render_markdown_table",
    "split_sentences",
    "is_table_row",
    "is_table_separator",
    "detect_table_fragmentation",
    "BATCH_UNIT_SIZE",
    "MAX_TABLE_TOKENS",
    "TARGET_GROUP_MAX_TOKENS",
    "NARRATIVE_SPLIT_TOKENS",
    "NARRATIVE_OVERLAP_TOKENS",
    "MAX_PARENT_TOKENS",
    "COMPANY_SPECS",
    "OUTPUT_CHUNKING_ROOT",
    "OUTPUT_PARSING_ROOT",
    "ENCODING_NAME",
    "FOOTNOTE_PATTERN",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "is_top_level_heading",
    "is_major_subsection",
    "update_heading_stack",
    "is_noise_or_fragment",
    "request_llm_semantic_grouping",
    "split_into_safe_batches",
    "fallback_heuristic_grouping",
    "extract_structural_units",
    "consolidate_semantic_groups",
    "build_dual_representation_chunk",
    "_make_narrative_chunk",
    "build_parent_chunks",
    "create_specialized_financial_chunks",
    "run_method5_for_company",
    "main",
]

if __name__ == "__main__":
    main()
