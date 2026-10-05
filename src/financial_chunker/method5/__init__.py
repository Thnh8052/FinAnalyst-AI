"""
Method 5: Proposed Golden Hybrid Financial Chunker Package

Package module hóa kiến trúc Parent-Child, Structure-Aware & LLM-Assisted Financial Chunking.
"""

from financial_chunker.method5.common import (
    BATCH_UNIT_SIZE,
    COMPANY_SPECS,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    detect_table_fragmentation,
    ENCODING_NAME,
    FOOTNOTE_PATTERN,
    is_table_row,
    is_table_separator,
    MAX_PARENT_TOKENS,
    MAX_TABLE_TOKENS,
    NARRATIVE_OVERLAP_TOKENS,
    NARRATIVE_SPLIT_TOKENS,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_PARSING_ROOT,
    render_markdown_table,
    split_sentences,
    StructuralUnit,
    TARGET_GROUP_MAX_TOKENS,
)
from financial_chunker.method5.headings import (
    is_major_subsection,
    is_noise_or_fragment,
    is_top_level_heading,
    update_heading_stack,
)
from financial_chunker.method5.grouper import (
    fallback_heuristic_grouping,
    request_llm_semantic_grouping,
    split_into_safe_batches,
)
from financial_chunker.method5.consolidator import (
    consolidate_semantic_groups,
    extract_structural_units,
)
from financial_chunker.method5.child_builder import (
    _make_narrative_chunk,
    build_dual_representation_chunk,
)
from financial_chunker.method5.parent_builder import (
    build_parent_chunks,
    create_specialized_financial_chunks,
)
from financial_chunker.method5.pipeline import (
    main,
    run_method5_for_company,
)

__all__ = [
    # Common & Models
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
    # Headings
    "is_top_level_heading",
    "is_major_subsection",
    "update_heading_stack",
    "is_noise_or_fragment",
    # Grouper
    "request_llm_semantic_grouping",
    "split_into_safe_batches",
    "fallback_heuristic_grouping",
    # Consolidator
    "extract_structural_units",
    "consolidate_semantic_groups",
    # Child Builder
    "build_dual_representation_chunk",
    "_make_narrative_chunk",
    # Parent Builder
    "build_parent_chunks",
    "create_specialized_financial_chunks",
    # Pipeline
    "run_method5_for_company",
    "main",
]
