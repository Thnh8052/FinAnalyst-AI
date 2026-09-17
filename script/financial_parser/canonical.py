"""Canonical Page JSON generation: the stable contract for structure chunking."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .markdown_utils import extract_markdown_tables
from .models import PageProfile, QCResult, RouteDecision


SECTION_RE = re.compile(
    r"^(?:(?:note|thuyết\s+minh|item|part)\s*)?([0-9ivx]+[a-z]?(?:\.[0-9a-z]+)*)[.\-:–—)]*\s+(.+)$",
    re.IGNORECASE,
)
NOTE_RE = re.compile(
    r"^(note|thuyết\s+minh|item|part)\s+([0-9ivx]+[a-z]?(?:\.[0-9a-z]+)*)(?:[.\-:–—)]*\s+)?(.+)?$",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+20\d{2}|20\d{2})\b", re.IGNORECASE)


def _unit_and_period(markdown: str) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
    lower = markdown.lower()
    unit_match = re.search(
        r"\(?(?:in\s+(?:millions?|thousands?|billions?)|amounts?\s+in|đơn\s+vị\s+tính\s*[:\-]?)[^\n\r\)]*\)?",
        markdown,
        re.IGNORECASE,
    )
    unit = None
    if unit_match:
        raw = unit_match.group(0).strip()
        if raw.startswith("(") and not raw.endswith(")"):
            raw += ")"
        elif raw.endswith(")") and not raw.startswith("("):
            raw = "(" + raw
        currency = "VND" if any(value in lower for value in ("vnd", "vnđ", "đồng")) else "USD" if any(value in lower for value in ("usd", "us$", "$", "form 10-k", "form 10-q", "sec filing")) else None
        scale = "billion" if any(value in lower for value in ("tỷ", "billion")) else "million" if any(value in lower for value in ("triệu", "million")) else "thousand" if any(value in lower for value in ("nghìn", "thousand")) else "unit"
        unit = {"raw": raw, "currency": currency, "scale": scale}
    dates = DATE_RE.findall(markdown)
    period = {"observed_dates": dates[:8]} if dates else None
    return unit, period


def _heading_metadata(value: str) -> Tuple[Optional[str], Optional[str]]:
    plain = value.strip().strip("*").strip()
    note = NOTE_RE.match(plain)
    if note:
        code = note.group(2)
        title_suffix = (note.group(3) or "").strip().lstrip(".-:–—) ").strip()
        title = f"{note.group(1)} {code} {title_suffix}".strip()
        return code, title
    section = SECTION_RE.match(plain)
    if section:
        return section.group(1), section.group(2).strip()
    return None, None



def markdown_to_canonical_page(
    *,
    markdown: str,
    document_id: str,
    pdf_page: int,
    printed_page: Optional[int],
    profile: PageProfile,
    route: RouteDecision,
    qc: QCResult,
) -> Dict[str, Any]:
    """Transform Markdown to blocks without guessing context from other pages."""
    tables = extract_markdown_tables(markdown)
    tables_by_line = {table["start_line"]: table for table in tables}
    lines = markdown.splitlines()
    blocks: List[Dict[str, Any]] = []
    current_section: Optional[str] = None
    section_codes: List[str] = []
    block_number = 1
    table_number = 1
    index = 0

    unit, period = _unit_and_period(markdown)
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        line_number = index + 1
        if not stripped:
            index += 1
            continue

        table = tables_by_line.get(line_number)
        if table:
            blocks.append({
                "block_id": f"p{pdf_page}_b{block_number:02d}",
                "block_type": "table",
                "table_id": f"p{pdf_page}_t{table_number:02d}",
                "parent_section_code": current_section,
                "section_resolution": "observed_on_page" if current_section else "unresolved",
                "headers": table["headers"],
                "rows": table["rows"],
                "unit": unit,
                "period": period,
                "source_refs": [{"pdf_page": pdf_page}],
            })
            block_number += 1
            table_number += 1
            index += 2 + len(table["rows"])
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            raw_heading = heading_match.group(2).strip()
            unit_suffix_match = re.search(
                r"\s*(\(?(?:in\s+(?:millions?|thousands?|billions?)|đơn\s+vị\s+tính\s*[:\-]?)[^\)\n]*\)?)\s*$",
                raw_heading,
                re.IGNORECASE,
            )
            if unit_suffix_match:
                unit_text = unit_suffix_match.group(1).strip()
                clean_heading = raw_heading[:unit_suffix_match.start()].strip()
                if clean_heading:
                    section_code, section_title = _heading_metadata(clean_heading)
                    if section_code:
                        current_section = section_code
                        if section_code not in section_codes:
                            section_codes.append(section_code)
                    blocks.append({
                        "block_id": f"p{pdf_page}_b{block_number:02d}",
                        "block_type": "heading",
                        "level": len(heading_match.group(1)),
                        "section_code": section_code,
                        "section_title": section_title or clean_heading,
                        "text": clean_heading,
                        "source_refs": [{"pdf_page": pdf_page}],
                    })
                    block_number += 1
                blocks.append({
                    "block_id": f"p{pdf_page}_b{block_number:02d}",
                    "block_type": "paragraph",
                    "parent_section_code": current_section,
                    "section_resolution": "observed_on_page" if current_section else "unresolved",
                    "text": unit_text,
                    "source_refs": [{"pdf_page": pdf_page}],
                })
            else:
                section_code, section_title = _heading_metadata(raw_heading)
                if section_code:
                    current_section = section_code
                    if section_code not in section_codes:
                        section_codes.append(section_code)
                blocks.append({
                    "block_id": f"p{pdf_page}_b{block_number:02d}",
                    "block_type": "heading",
                    "level": len(heading_match.group(1)),
                    "section_code": section_code,
                    "section_title": section_title or raw_heading,
                    "text": raw_heading,
                    "source_refs": [{"pdf_page": pdf_page}],
                })
        else:
            blocks.append({
                "block_id": f"p{pdf_page}_b{block_number:02d}",
                "block_type": "paragraph",
                "parent_section_code": current_section,
                "section_resolution": "observed_on_page" if current_section else "unresolved",
                "text": stripped,
                "source_refs": [{"pdf_page": pdf_page}],
            })
        block_number += 1
        index += 1

    return {
        "schema_version": "financial-parser-page-v1",
        "page": {
            "document_id": document_id,
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "language": profile.language.to_dict(),
            "page_profile": profile.to_dict(include_raw_text=False),
            "route": route.to_dict(),
            "qc": qc.to_dict(),
            "section_codes": section_codes,
            "unit": unit,
            "period": period,
            "has_tables": bool(tables),
            "table_count": len(tables),
            "blocks": blocks,
        },
    }
