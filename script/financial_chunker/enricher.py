"""
enricher.py

MetadataEnricher:
- Nhận diện statement_type với bảng ưu tiên 8 cấp nghiêm ngặt (comprehensive_income trước income_statement).
- Chuẩn hóa ngày tài chính sang định dạng ISO 8601 (YYYY-MM-DD).
- Nhận diện đơn vị tiền tệ (USD, VND, EUR) và quy mô (million, billion, thousand).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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


def parse_iso_date(raw_date_str: str) -> Optional[str]:
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

    # 4. Chỉ có năm (vd: 2025)
    m_year = re.search(r"\b(20\d{2})\b", s)
    if m_year:
        return f"{m_year.group(1)}-12-31"

    return None


def extract_iso_dates_from_period(period_data: Any) -> List[str]:
    if not period_data:
        return []

    dates: List[str] = []
    if isinstance(period_data, dict):
        observed = period_data.get("observed_dates", [])
        if isinstance(observed, list):
            for d in observed:
                iso = parse_iso_date(str(d))
                if iso and iso not in dates:
                    dates.append(iso)
        raw = period_data.get("raw", "")
        if raw:
            for part in re.split(r"[,;vs\-\–]", str(raw)):
                iso = parse_iso_date(part)
                if iso and iso not in dates:
                    dates.append(iso)
    elif isinstance(period_data, list):
        for item in period_data:
            iso = parse_iso_date(str(item))
            if iso and iso not in dates:
                dates.append(iso)
    elif isinstance(period_data, str):
        iso = parse_iso_date(period_data)
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
