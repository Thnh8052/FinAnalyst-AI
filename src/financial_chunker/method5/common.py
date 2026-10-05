"""
common.py - Method 5 (Proposed Golden Hybrid) Shared Constants, Models, and Formatting Utilities
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv

from financial_chunker.config import COMPANY_SPECS, OUTPUT_CHUNKING_ROOT, OUTPUT_PARSING_ROOT
from financial_chunker.models import StitchedTable

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = "deepseek-chat"

ENCODING_NAME = "cl100k_base"
BATCH_UNIT_SIZE = 16
MAX_TABLE_TOKENS = 1500
TARGET_GROUP_MAX_TOKENS = 1100
NARRATIVE_SPLIT_TOKENS = 500
NARRATIVE_OVERLAP_TOKENS = 75
MAX_PARENT_TOKENS = 2200

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
    heading_path: List[str] = field(default_factory=list)
    table_obj: Optional[StitchedTable] = None
    qc_status: str = "pass"


def _esc_table_cell(s: Any) -> str:
    """Escape backslash và ký tự pipe '|' để không phá vỡ bảng Markdown."""
    clean = str(s).replace("\n", " ").strip()
    clean = clean.replace("\\", "\\\\").replace("|", "\\|")
    return clean


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render headers và rows dạng ma trận bảng 2D Markdown an toàn."""
    if not headers and not rows:
        return ""

    num_cols = len(headers) if headers else (len(rows[0]) if rows else 1)
    clean_headers = [_esc_table_cell(h) for h in headers]
    while len(clean_headers) < num_cols:
        clean_headers.append(f"Col {len(clean_headers)+1}")

    header_line = "| " + " | ".join(clean_headers) + " |"
    separator_line = "| " + " | ".join(["---"] * num_cols) + " |"

    rendered_rows = []
    for row in rows:
        clean_cells = [_esc_table_cell(c) for c in row]
        while len(clean_cells) < num_cols:
            clean_cells.append("")
        rendered_rows.append("| " + " | ".join(clean_cells[:num_cols]) + " |")

    return "\n".join([header_line, separator_line] + rendered_rows)


def split_sentences(text: str) -> List[str]:
    """Tách câu tiếng Anh/Việt theo dấu chấm câu an toàn."""
    sentences = re.split(r"(?<!\d\.)(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def is_table_row(line: str) -> bool:
    """True nếu chuỗi là một dòng của bảng Markdown."""
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def is_table_separator(line: str) -> bool:
    """True nếu chuỗi là dòng phân cách giữa header và data trong bảng Markdown."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return all(re.match(r"^:?-+:?$", c) for c in cells if c)


def detect_table_fragmentation(chunk_text: str) -> Tuple[bool, List[str]]:
    """Phát hiện lỗi vỡ bảng (thiếu header hoặc bắt đầu giữa chừng) trong chunk text."""
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
