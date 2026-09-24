"""
enricher.py

MetadataEnricher:
- Nhận diện statement_type với bảng ưu tiên nghiêm ngặt (comprehensive_income trước income_statement).
- Chuẩn hóa ngày tài chính sang định dạng ISO 8601 (YYYY-MM-DD).
- Nhận diện đơn vị tiền tệ (USD, VND, EUR) và quy mô (million, billion, thousand).
- Phân loại 2 cờ boolean (contains_table, contains_text) với ngưỡng 30 từ, đảm bảo 0 ghost chunks (F, F).

Schema Specification (Freeze V0):
- SCHEMA_VERSION: "v0.1.0-frozen-2026-09" (Frozen at 2026-09-23)
- TICKERS_V0: ["AAPL", "NVDA", "AMZN", "AMD", "INTC", "NKE", "WMT"] (7 tập đoàn SEC 10-K)
- FISCAL_PERIODS: ["FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"] (V0 chỉ dùng "FY", không ghép năm)
- FORM_TYPES: ["10-K", "10-Q", "8-K", "20-F"] (V0 chỉ dùng "10-K")
- STATEMENT_TYPES: 9 loại chuẩn (income_statement, balance_sheet, cash_flows, comprehensive_income, equity, notes, mda, risk_factors, other_financial)

Ghi chú về has_table_fragmentation:
- Ý nghĩa: Cờ đánh giá chất lượng phân mảnh bảng (Table Fragmentation Metric). Đánh dấu True nếu chunk bị cắt ngang một bảng số liệu mà thiếu dòng tiêu đề / cột (header).
- Thiết lập bởi: Các module Chunker (Method 1, 3, 4, 5). Method 5 đạt tỷ lệ 0% nhờ kiến trúc giữ trọn vẹn bảng.
- Kiểu dữ liệu: bool (mặc định False).
- Phạm vi sử dụng: Chỉ số đánh giá so sánh chất lượng chunking cho Báo cáo ĐATN, lưu trong Provenance Payload, KHÔNG ĐÁNH INDEX và không dùng làm bộ lọc tìm kiếm cho người dùng cuối.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ============================================================
# SCHEMA CONSTANTS & VERSIONING (FREEZE V0)
# ============================================================
SCHEMA_VERSION = "v0.1.0-frozen-2026-09"
SCHEMA_FROZEN_AT = "2026-09-23"

CONTAINS_TEXT_MIN_WORDS = 30
CONTAINS_TABLE_MIN_TABLE_LINES = 2
WORD_COUNT_METHOD = "whitespace_split"

EXCLUDED_PREFIXES = [
    "[Document:",
    "[Statement Type:",
    "[Unit:",
    "[Period Dates:",
]

# Chu kỳ tài chính chuẩn hoá (Period) - V0 độc lập không ghép năm
FISCAL_PERIODS = ["FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2"]
FORM_TYPES = ["10-K", "10-Q", "8-K", "20-F"]
TICKERS_V0 = ["AAPL", "NVDA", "AMZN", "AMD", "INTC", "NKE", "WMT"]

MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

STATEMENT_PRIORITY_PATTERNS = [
    ("cash_flows", re.compile(
        r"(?:consolidated\s+)?statements?\s+of\s+cash\s+flows?", re.I)),
    ("balance_sheet", re.compile(
        r"(?:consolidated\s+)?(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)", re.I)),
    ("comprehensive_income", re.compile(
        r"(?:consolidated\s+)?statements?\s+of\s+(?:comprehensive\s+(?:income|loss)|other\s+comprehensive\s+income)", re.I)),
    ("equity", re.compile(
        r"(?:consolidated\s+)?statements?\s+of\s+(?:stockholders?['’]?\s+equity|shareholders?['’]?\s+equity|changes\s+in\s+equity)", re.I)),
    ("income_statement", re.compile(
        r"(?:consolidated\s+)?statements?\s+of\s+(?:income|operations|earnings|profit\s+and\s+loss|loss)", re.I)),
    ("mda", re.compile(r"(?:item\s+7\.?\s+)?management['’]?s\s+discussion", re.I)),
    ("risk_factors", re.compile(r"(?:item\s+1a\.?\s+)?risk\s+factors", re.I)),
    ("notes", re.compile(r"(?:notes?\s+to\s+consolidated\s+financial\s+statements?|note\s+\d+)", re.I)),
]


def detect_statement_type(title: str, text_sample: str = "") -> str:
    combined_title = title or ""
    for stmt_type, pattern in STATEMENT_PRIORITY_PATTERNS:
        if pattern.search(combined_title):
            return stmt_type

    combined_sample = text_sample[:300] if text_sample else ""
    for stmt_type, pattern in STATEMENT_PRIORITY_PATTERNS[:5]:
        if pattern.search(combined_sample):
            return stmt_type

    return "other_financial"


# Bảng fallback tĩnh dự phòng cuối cùng nếu văn bản không chứa bất kỳ thông tin cover page nào
FISCAL_YEAR_END_MAP = {
    "NVDA": "-01-31",
    "NVIDIA": "-01-31",
    "WMT": "-01-31",
    "WALMART": "-01-31",
    "NKE": "-05-31",
    "NIKE": "-05-31",
    "AAPL": "-09-30",
    "APPLE": "-09-30",
    "AMZN": "-12-31",
    "AMAZON": "-12-31",
    "AMD": "-12-31",
    "INTC": "-12-31",
    "INTEL": "-12-31",
}

COVER_FISCAL_DATE_PATTERN = re.compile(
    r"for the (?:fiscal\s+year|quarterly\s+period|period)\s+ended\s+([A-Za-z]+)\s+([0-9]{1,2}),?\s*(20\d{2})?",
    re.I,
)

STATEMENT_FISCAL_DATE_PATTERN = re.compile(
    r"(?:years?|period)\s+ended\s+([A-Za-z]+)\s+([0-9]{1,2}),?\s*(20\d{2})?",
    re.I,
)


def detect_document_fiscal_calendar(pages_data: List[Dict[str, Any]]) -> Optional[str]:
    """
    Tự động phát hiện ngày kết thúc năm tài chính (fiscal year end) động từ trang bìa
    SEC Form 10-K mà KHÔNG CẦN HARDCODE bất kỳ Ticker hay Tên công ty nào.

    Theo luật SEC (Exchange Act of 1934), 100% hồ sơ 10-K bắt buộc phải có dòng:
    'For the fiscal year ended [Month Day, Year]' trên Trang bìa.

    Returns:
        Suffix định dạng '-MM-DD' (vd: '-09-28' cho Apple 2024, '-01-26' cho Nvidia 2025),
        hoặc None nếu không tìm thấy.
    """
    if not pages_data:
        return None

    # Tầng 1: Quét tối đa 4 trang đầu tiên (Trang bìa SEC 10-K)
    for p_data in pages_data[:4]:
        text_content = ""
        page_obj = p_data.get("page", p_data)
        blocks = page_obj.get("blocks", [])
        if blocks:
            text_content = " ".join([str(b.get("text", "")) for b in blocks])
        else:
            text_content = page_obj.get("text", "") or json.dumps(page_obj, ensure_ascii=False)

        clean_text = text_content.replace("\xa0", " ").replace("\\xa0", " ")
        m = COVER_FISCAL_DATE_PATTERN.search(clean_text)
        if m:
            month_str = m.group(1).lower()
            if month_str in MONTH_MAP:
                m_num = MONTH_MAP[month_str]
                d_num = f"{int(m.group(2)):02d}"
                return f"-{m_num}-{d_num}"

    # Tầng 2: Quét tiêu đề Báo cáo tài chính (Financial Statement Headers)
    for p_data in pages_data[:15]:
        page_obj = p_data.get("page", p_data)
        blocks = page_obj.get("blocks", [])
        for b in blocks:
            b_text = str(b.get("text", "")).replace("\xa0", " ").replace("\\xa0", " ")
            m = STATEMENT_FISCAL_DATE_PATTERN.search(b_text)
            if m:
                month_str = m.group(1).lower()
                if month_str in MONTH_MAP:
                    m_num = MONTH_MAP[month_str]
                    d_num = f"{int(m.group(2)):02d}"
                    return f"-{m_num}-{d_num}"

    return None


def parse_iso_date(
    raw_date_str: str,
    ticker: Optional[str] = None,
    fiscal_date_suffix: Optional[str] = None,
) -> Optional[str]:
    s = raw_date_str.strip()
    # 1. Định dạng YYYY-MM-DD
    m_iso = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", s)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # 2. Định dạng Month Day, Year (vd: Jan 26, 2025 hoặc January 26, 2025)
    m_text = re.search(
        r"\b([A-Za-z]+)\s+([0-9]{1,2}),\s*(20\d{2})\b", s
    )
    if m_text:
        month_name = m_text.group(1).lower()
        if month_name in MONTH_MAP:
            m_num = MONTH_MAP[month_name]
            d_num = f"{int(m_text.group(2)):02d}"
            y_num = m_text.group(3)
            return f"{y_num}-{m_num}-{d_num}"

    # 3. Định dạng MM/DD/YYYY
    m_slash = re.search(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(20\d{2})\b", s)
    if m_slash:
        m_num = f"{int(m_slash.group(1)):02d}"
        d_num = f"{int(m_slash.group(2)):02d}"
        y_num = m_slash.group(3)
        return f"{y_num}-{m_num}-{d_num}"

    # 4. Chỉ có năm (vd: 2025 hoặc FY2025) -> Dùng fiscal_date_suffix động, fallback ticker map, cuối cùng là -12-31
    m_year = re.search(r"\b(?:FY)?(20\d{2})\b", s, re.I)
    if m_year:
        year_str = m_year.group(1)
        if fiscal_date_suffix:
            fy_suffix = fiscal_date_suffix
        elif ticker:
            fy_suffix = FISCAL_YEAR_END_MAP.get(ticker.upper(), "-12-31")
        else:
            fy_suffix = "-12-31"
        return f"{year_str}{fy_suffix}"

    return None


def extract_iso_dates_from_period(
    period_data: Any,
    ticker: Optional[str] = None,
    fiscal_date_suffix: Optional[str] = None,
) -> List[str]:
    if not period_data:
        return []

    dates: List[str] = []
    if isinstance(period_data, dict):
        observed = period_data.get("observed_dates", [])
        if isinstance(observed, list):
            for d in observed:
                iso = parse_iso_date(str(d), ticker=ticker, fiscal_date_suffix=fiscal_date_suffix)
                if iso and iso not in dates:
                    dates.append(iso)
        raw = period_data.get("raw", "")
        if raw:
            for part in re.split(r"[,;vs\-\–]", str(raw)):
                iso = parse_iso_date(part, ticker=ticker, fiscal_date_suffix=fiscal_date_suffix)
                if iso and iso not in dates:
                    dates.append(iso)
    elif isinstance(period_data, list):
        for item in period_data:
            iso = parse_iso_date(str(item), ticker=ticker, fiscal_date_suffix=fiscal_date_suffix)
            if iso and iso not in dates:
                dates.append(iso)
    elif isinstance(period_data, str):
        iso = parse_iso_date(period_data, ticker=ticker, fiscal_date_suffix=fiscal_date_suffix)
        if iso:
            dates.append(iso)

    return sorted(dates)



def parse_unit_metadata(unit_data: Any) -> Dict[str, str]:
    currency = "USD"
    scale = "million"
    raw_str = ""

    if isinstance(unit_data, dict):
        raw_str = str(unit_data.get("raw", ""))
        currency = unit_data.get("currency") or "USD"
        scale = unit_data.get("scale") or "million"
    elif isinstance(unit_data, str):
        raw_str = unit_data

    raw_lower = raw_str.lower()
    if "vnd" in raw_lower or "đồng" in raw_lower:
        currency = "VND"
    elif "eur" in raw_lower or "euro" in raw_lower:
        currency = "EUR"
    elif "usd" in raw_lower or "$" in raw_lower:
        currency = "USD"

    if "billion" in raw_lower or "tỷ" in raw_lower:
        scale = "billion"
    elif "million" in raw_lower or "triệu" in raw_lower:
        scale = "million"
    elif "thousand" in raw_lower or "nghìn" in raw_lower:
        scale = "thousand"

    return {"currency": currency, "scale": scale, "raw": raw_str}


def compute_content_flags(
    chunk_text: str,
    chunk_type: str = "",
    min_table_lines: int = CONTAINS_TABLE_MIN_TABLE_LINES,
    narrative_word_threshold: int = CONTAINS_TEXT_MIN_WORDS,
    excluded_prefixes: List[str] = EXCLUDED_PREFIXES,
) -> tuple[bool, bool]:
    """
    Xác định 2 cờ boolean (contains_table, contains_text) cho Financial Chunks:
    
    Quy tắc nghiệp vụ:
    1. contains_table:
       - True nếu chunk_type thuộc nhóm bảng ('table_atomic', 'table_subgroup', 'table')
         hoặc có ít nhất min_table_lines (mặc định 2) dòng markdown table bắt đầu bằng '|'.
       - False nếu không chứa cấu trúc bảng.
       
    2. contains_text:
       - Nếu contains_table == False: Toàn bộ chunk là văn bản tự sự (narrative/heading/disclosure).
         -> contains_text = True (ngăn chặn hiện tượng 'ghost chunk' (False, False)).
       - Nếu contains_table == True:
         -> Loại bỏ dòng cấu trúc bảng Markdown ('|...|', '+---', '|---').
         -> Loại bỏ metadata prefix do hệ thống gắn (theo danh sách EXCLUDED_PREFIXES).
         -> Đếm số từ tự sự bằng whitespace split: len(narrative_text.split()).
         -> contains_text = (narrative_word_count >= narrative_word_threshold, mặc định 30 từ).
         
    Kết quả:
    - (True, False) : Bảng thuần túy (Pure Table) -> Lọc trúng các báo cáo tài chính lớn
    - (True, True)  : Bảng hỗn hợp (Mixed Table + Narrative)
    - (False, True) : Văn bản tự sự thuần túy (Pure Narrative)
    - (False, False): Tuyệt đối không xảy ra với dữ liệu hợp lệ (0 chunks)
    """
    lines = chunk_text.splitlines() if chunk_text else []
    
    # 1. Nhận diện cấu trúc bảng
    table_lines = [l for l in lines if l.strip().startswith("|")]
    is_table_type = chunk_type in ("table_atomic", "table_subgroup", "table")
    contains_table = is_table_type or (len(table_lines) >= min_table_lines)
    
    # 2. Xử lý contains_text
    if not contains_table:
        # Khi không có bảng, chunk luôn là văn bản tự sự
        return False, True
        
    # Trường hợp có bảng: Lọc bỏ dòng bảng, dòng separator, dòng metadata hệ thống
    text_lines = [
        l for l in lines
        if not l.strip().startswith("|")
        and not any(l.strip().startswith(p) for p in excluded_prefixes)
        and not re.match(r"^\s*[+|\-]{3,}", l.strip())
    ]
    
    narrative_text = " ".join(text_lines).strip()
    word_count = len(narrative_text.split())
    contains_text = word_count >= narrative_word_threshold
    
    return True, contains_text
