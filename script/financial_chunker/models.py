"""
models.py

Định nghĩa cấu trúc dữ liệu chuẩn cho toàn bộ hệ thống Financial Chunker:
- ChunkMethod, ChunkType
- FinancialChunk, ChunkMetadata
- StitchedTable, StitchedSection, StitchedDocument
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChunkMethod(str, Enum):
    METHOD1_FIXED_SIZE = "method1_fixed_size"
    METHOD2_DETERMINISTIC = "method2_deterministic"
    METHOD2_5_HEADING_LLM = "method2_5_heading_llm"
    METHOD3_LLM_ASSISTED = "method3_llm_assisted"
    METHOD4_BOUNDARY_TAGGING = "method4_boundary_tagging"
    METHOD5_GOLDEN_HYBRID = "method5_golden_hybrid"


class ChunkType(str, Enum):
    FIXED_SIZE = "fixed_size"
    TABLE_ATOMIC = "table_atomic"
    TABLE_SUBGROUP = "table_subgroup"
    NARRATIVE_SECTION = "narrative_section"
    DOCUMENT_SUMMARY = "document_summary"
    CROSS_REFERENCE_INDEX = "cross_reference_index"


@dataclass
class StitchedTable:
    logical_table_id: str
    source_pages: List[int]
    source_table_ids: List[str]
    source_block_ids: List[str]
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    unit: Optional[Dict[str, str]] = None
    period: Optional[Dict[str, Any]] = None
    footnotes: List[str] = field(default_factory=list)
    qc_status: str = "pass"
    is_multi_page: bool = False


@dataclass
class StitchedSection:
    logical_section_id: str
    section_code: Optional[str]
    section_title: str
    parent_section_id: Optional[str]
    source_pages: List[int]
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    narrative_text: str = ""


@dataclass
class StitchedDocument:
    document_id: str
    ticker: str
    fiscal_year: int
    tables: List[StitchedTable] = field(default_factory=list)
    sections: List[StitchedSection] = field(default_factory=list)
    cross_page_links: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FinancialChunk:
    chunk_id: str
    document_id: str
    ticker: str
    fiscal_year: int
    chunk_method: str
    chunk_type: str
    content: str
    content_retrieval: str
    content_generation: str
    token_count: int
    source_pages: List[int]
    has_table_fragmentation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
