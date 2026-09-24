"""SCHEMA_FROZEN.py / schema.py

Đóng băng chuẩn Schema (Freeze V0 Specification) cho FinAnalyst-AI Qdrant Vector Database.
Đảm bảo tính nhất quán tuyệt đối giữa:
- Parsing/Chunking MetadataEnricher (financial_chunker.enricher)
- Vector Indexing & Qdrant Payload (financial_rag.indexing)
- Retrieval Filtering & Query Routing (financial_rag.retrieval)
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ============================================================
# SCHEMA VERSIONING & FREEZE METADATA
# ============================================================
SCHEMA_VERSION = "v0.1.1-frozen-2026-09"
SCHEMA_FROZEN_AT = "2026-09-23"

# ============================================================
# NGƯỠNG PHÂN LOẠI NỘI DUNG (FREEZE V0)
# ============================================================
CONTAINS_TEXT_MIN_WORDS = 30
CONTAINS_TABLE_MIN_TABLE_LINES = 2
WORD_COUNT_METHOD = "whitespace_split"

EXCLUDED_PREFIXES: List[str] = [
    "[Document:",
    "[Statement Type:",
    "[Unit:",
    "[Period Dates:",
]

# ============================================================
# PAYLOAD INDEXES — FREEZE V0
# ============================================================
# FIX [S1]: Xóa comment duplicate (7 vs 8 fields). Chuẩn hóa thành 9 fields.
# FIX [S2]: Thêm "period_years" để hỗ trợ L1/L2 filter khi filing chứa data nhiều năm.
#           ⚠️ Yêu cầu RE-INDEX Qdrant collection nếu dùng period_years filter.
PAYLOAD_INDEXES_V0: Dict[str, str] = {
    "ticker": "keyword",             # AAPL, NVDA, AMZN, AMD, INTC, NKE, WMT
    "document_id": "keyword",        # e.g. "nvidia_2025_10k"
    "fiscal_year": "integer",        # Filing year (2021..2025)
    "period_years": "integer[]",     # Data years trong chunk [2025, 2024, 2023]
    "fiscal_period": "keyword",      # "FY" | "Q1".."Q4" | "H1", "H2"
    "statement_type": "keyword",     # 9 giá trị chuẩn
    "form_type": "keyword",          # "10-K" | "10-Q", "8-K", "20-F"
    "contains_table": "bool",
    "contains_text": "bool",
}

# ============================================================
# SIMPLIFIED METADATA SPECIFICATION (SEC-INSIGHTS ALIGNED)
# ============================================================
# Tinh gọn siêu dữ liệu theo triết lý sec-insights kết hợp bảo toàn mảng period_years:
# 1. Core Document Fields: ticker, document_id, fiscal_year, form_type, source_pages
# 2. Financial Multi-Year Extension (Bảo toàn): period_years (mảng số nguyên cho so sánh liên kỳ & năm quá khứ)
# 3. Relaxed Optional Fields: statement_type, fiscal_period, contains_table, contains_text (không ép buộc lọc)
SIMPLIFIED_METADATA_FIELDS: List[str] = [
    "ticker",
    "document_id",
    "fiscal_year",
    "period_years",
    "form_type",
    "source_pages",
]
SIMPLIFIED_FILTER_DEFAULT: bool = True

# ============================================================
# PROVENANCE — KHÔNG INDEX
# ============================================================
PROVENANCE_FIELDS: List[str] = [
    "chunk_id",
    "company",
    "section_title",
    "period_dates",
    "source_pages",
    "currency",
    "unit",
    "token_count",
    "has_fact_tuples",
    "chunk_method",
    "chunk_type",
    "embedding_model",
    "embedding_version",
    "has_table_fragmentation",
]

# ============================================================
# CONTENT — KHÔNG INDEX
# ============================================================
CONTENT_FIELDS: List[str] = [
    "content_retrieval",
    "content_generation",
    "content",
]

# ============================================================
# ENUM CONSTANTS
# ============================================================
TICKERS_V0: List[str] = ["AAPL", "NVDA", "AMZN", "AMD", "INTC", "NKE", "WMT"]

STATEMENT_TYPES: List[str] = [
    "income_statement",
    "balance_sheet",
    "cash_flows",
    "comprehensive_income",
    "equity",
    "notes",
    "mda",
    "risk_factors",
    "other_financial",
]

FISCAL_PERIODS: List[str] = ["FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"]

FORM_TYPES: List[str] = ["10-K", "10-Q", "8-K", "20-F"]

# ============================================================
# GOLD BENCHMARK SPECIFICATION
# ============================================================
ALL_LEVELS: List[str] = ["L1", "L2", "L3", "L4", "L5"]
RETRIEVAL_LEVELS: List[str] = ["L1", "L2", "L3", "L4"]
ABSTENTION_LEVEL: str = "L5"


@dataclass(frozen=True)
class LevelThresholds:
    """Ngưỡng đánh giá phân tầng chi tiết theo Level."""
    numbers_required: float
    keywords_required: float
    containment_min: float


LEVEL_THRESHOLDS: Dict[str, LevelThresholds] = {
    "L1": LevelThresholds(numbers_required=1.00, keywords_required=0.60, containment_min=0.40),
    "L2": LevelThresholds(numbers_required=1.00, keywords_required=0.60, containment_min=0.50),
    "L3": LevelThresholds(numbers_required=0.75, keywords_required=0.60, containment_min=0.45),
    "L4": LevelThresholds(numbers_required=0.00, keywords_required=0.60, containment_min=0.35),
    "L5": LevelThresholds(numbers_required=0.00, keywords_required=0.00, containment_min=0.00),
}

# Backward compatibility
CONTAINMENT_THRESHOLDS: Dict[str, float] = {
    k: v.containment_min for k, v in LEVEL_THRESHOLDS.items()
}

# Đơn vị tài chính
UNIT_MULTIPLIERS: Dict[str, float] = {
    "raw": 1e-6,
    "dollar": 1e-6,
    "usd": 1e-6,
    "thousand": 1e-3,
    "k": 1e-3,
    "million": 1.0,
    "m": 1.0,
    "mn": 1.0,
    "mm": 1.0,
    "billion": 1e3,
    "b": 1e3,
    "bn": 1e3,
    "trillion": 1e6,
    "t": 1e6,
    "percent": 1.0,
    "pct": 1.0,
    "%": 1.0,
}

# Whitelist entities
WHITELIST_PATTERNS: List[str] = [
    r"\bQ[1-4]\b",
    r"\bNotes?\s+\d+[A-Z]?\b",
    r"\bItem\s+\d+[A-Z]?\b",
    r"\bFY\s*\d{2,4}\b",
    r"\b\d+(?:\.\d+)?[A-Za-z]\b",
    r"\b\d+(?:\.\d+)?nm\b",
    r"\bSection\s+\d+(?:\(\w+\))?\b",
]

# Token regex giữ số có separator và magnitude suffix
TOKEN_REGEX: str = r"\b[\w]+(?:[.,]\d+)*[A-Za-z]?\b"

# FIX [S8]: Word-boundary pattern thay vì substring check (tránh "periodic" match "period").
_YEAR_CONTEXT_PATTERN = re.compile(
    r"\b(fy|fiscal|year|ended|dated|annual|năm|niên\s+độ)\b",
    re.IGNORECASE,
)


def _has_year_context(prev_ctx: str = "", next_ctx: str = "") -> bool:
    """Check if surroundings indicate a calendar/fiscal year rather than a financial amount."""
    ctx = f"{prev_ctx} {next_ctx}"
    return bool(_YEAR_CONTEXT_PATTERN.search(ctx))


# FIX [S3]: Document rõ semantics của strip_magnitude_suffix.
#   - Gold facts: nên dùng strip_magnitude_suffix=True (vì value tách khỏi unit).
#   - Chunk text: giữ nguyên suffix (500M) vì đó là dữ liệu gốc.
#   - Khi so khớp: normalize cả 2 phía với CÙNG setting, hoặc match by value only.
def normalize_financial_number(
    s: str,
    prev_ctx: str = "",
    next_ctx: str = "",
    strip_magnitude_suffix: bool = False,
) -> str:
    """Normalize US accounting format uniformly on both Gold facts and Chunk text."""
    s = str(s).strip()
    # FIX: Hỗ trợ tiền tệ có khoảng trắng, ví dụ: "$ (1,234)" hoặc "($ 1,234)"
    is_paren = bool(re.match(r"^[$€£¥₫]?\s*\([^\)]+\)$|^\([$€£¥₫]?\s*[^\)]+\)$", s))
    has_currency = any(sym in s for sym in ["$", "€", "£", "¥", "₫"])
    s_clean = re.sub(r"[$€£¥₫(),% ]", "", s).strip()

    if is_paren and not has_currency:
        if (
            _has_year_context(prev_ctx, next_ctx)
            and s_clean.isdigit()
            and len(s_clean) == 4
            and 1990 <= int(s_clean) <= 2035
        ):
            return s_clean

    res = f"-{s_clean.lstrip('-')}" if (is_paren or s.strip().startswith("-")) else s_clean
    if strip_magnitude_suffix:
        res = re.sub(r"[KkMmBbTt]$", "", res)
    return res


def should_keep_number(num_str: str, prev_ctx: str = "", next_ctx: str = "") -> bool:
    """Keep number only if financial context is present and not a structural marker."""
    prev_lower = prev_ctx.lower()
    next_lower = next_ctx.lower()

    # 1. Financial symbols → keep immediately
    if any(sym in num_str or sym in prev_ctx or sym in next_ctx
           for sym in ["$", "€", "£", "¥", "₫", "%"]):
        return True

    # 2. Structural drop keywords
    drop_kw = ["page", "table", "figure", "exhibit", "chart", "trang", "bảng", "hình", "sơ đồ"]
    if any(kw in prev_lower for kw in drop_kw):
        return False
    if any(prev_lower.rstrip().endswith(kw) for kw in ["note", "notes", "item", "items"]):
        return False

    cleaned = num_str.lstrip("-").replace(",", "").replace(".", "").strip()
    n_digits = len(cleaned)
    if n_digits >= 3:
        return True
    if "." in num_str:
        return True
    if any(kw in prev_lower for kw in ["eps", "margin", "ratio", "growth", "rate"]):
        return True
    return False


def normalize_to_million(value: float, unit: str = "million") -> float:
    mult = UNIT_MULTIPLIERS.get(unit.lower().strip(), 1.0)
    return float(value) * mult


def normalize_numbers(text: str) -> str:
    """Normalize numeric strings in text (thousand separators, accounting parens)."""

    def replace_accounting_parens(match):
        val_str = match.group(1).replace(",", "").strip()
        start = match.start()
        end = match.end()
        prev_ctx = text[max(0, start - 30):start]
        next_ctx = text[end:min(len(text), end + 30)]
        if (
            _has_year_context(prev_ctx, next_ctx)
            and val_str.isdigit()
            and len(val_str) == 4
            and 1990 <= int(val_str) <= 2035
        ):
            return val_str
        return f"-{val_str}"

    # 1. Chuẩn hóa số âm dạng ngoặc đơn: (1,234) hoặc ($ 1,234) hoặc ( $ 1,234 )
    text = re.sub(
        r"\(\s*[$€£¥₫]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*\)",
        replace_accounting_parens,
        text,
    )
    # 2. FIX [B1]: Khử triệt để dấu phẩy hàng nghìn bằng Lookaround regex (xử lý số >= 2 dấu phẩy)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text


def extract_chunk_text(chunk_payload: Dict[str, Any]) -> str:
    return str(chunk_payload.get("content_retrieval") or chunk_payload.get("content") or "")


def compute_containment(evidence_text: str, chunk_text: str) -> float:
    clean_ev = normalize_numbers(evidence_text.lower())
    clean_ch = normalize_numbers(chunk_text.lower())

    ev_tokens = re.findall(TOKEN_REGEX, clean_ev)
    ch_tokens = re.findall(TOKEN_REGEX, clean_ch)

    if not ev_tokens:
        return 0.0

    ev_counter = Counter(ev_tokens)
    ch_counter = Counter(ch_tokens)
    overlap_count = sum((ev_counter & ch_counter).values())
    return overlap_count / sum(ev_counter.values())


def infer_expected_chunk_count(
    question_or_difficulty: Any,
    comparison_targets: Optional[List[Any]] = None,
    table_row_count: int = 1,
) -> int:
    if isinstance(question_or_difficulty, dict):
        q = question_or_difficulty
        level = q.get("difficulty", "L1")
        targets = q.get("comparison_targets") or []
        row_cnt = q.get("table_row_count", 1)
        q_text = str(q.get("question", "")).lower()
    else:
        level = str(question_or_difficulty)
        targets = comparison_targets or []
        row_cnt = table_row_count
        q_text = ""

    if level == "L1":
        return 1
    elif level == "L2":
        return 2 if row_cnt > 15 else 1
    elif level == "L3":
        return len(targets) if targets else 2
    elif level == "L4":
        multi_aspects = ["drivers", "factors", "reasons", "policies", "risks", "components", "initiatives"]
        if any(k in q_text for k in multi_aspects) or (targets and len(targets) > 1):
            return min(len(targets), 3) if targets else 2
        return 1
    elif level == "L5":
        return 0
    return 1
