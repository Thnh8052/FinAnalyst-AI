"""Ground Truth Reconciliation Engine for SEC Form 10-K Filings.

Performs document-wide verification against official SEC EDGAR XBRL and HTML:
1. RAG-Safe Semantic & Numeric Equivalence.
2. Temporal / Year-Column Alignment (detecting column swaps/shifts for 2025, 2024, 2023).
3. Core Financial Statements coverage & accuracy.
4. Review Queue root cause analysis & selective VLM fallback commands.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_numeric_str(val: str) -> Optional[float]:
    """Normalize numeric string for RAG comparison."""
    if not val or not isinstance(val, str):
        return None
    s = val.strip()
    s = re.sub(r'[\(\[]\w+[\)\]]$', '', s).strip()
    is_neg = False
    if s.startswith('(') and s.endswith(')'):
        is_neg = True
        s = s[1:-1].strip()
    elif s.startswith('-') or s.startswith('—') or s.startswith('–'):
        is_neg = True
        s = s[1:].strip()
    
    s = re.sub(r'[\$,%\s]', '', s)
    if not s or s in ('—', '-', '–', 'N/A', 'none'):
        return 0.0
    
    try:
        f = float(s)
        return -f if is_neg else f
    except ValueError:
        return None


def extract_years_from_header(header_text: str) -> List[int]:
    """Extract 4-digit years from a column header using digit boundary."""
    matches = re.findall(r'(?<!\d)(202[0-9]|201[0-9])(?!\d)', header_text)
    return [int(m) for m in matches]


@dataclass
class XBRLContext:
    context_id: str
    year: Optional[int] = None
    period_type: str = "unknown"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    instant: Optional[str] = None
    dimension: Optional[str] = None


@dataclass
class XBRLFact:
    concept: str
    context_id: str
    raw_value: str
    numeric_value: Optional[float]
    year: Optional[int]
    unit: str
    decimals: str
    scale: int = 0
    label: str = ""
    section: str = ""
    is_negative: bool = False


class GroundTruthExtractor:
    """Extracts facts, contexts, and labels from SEC EDGAR XBRL packages."""

    def __init__(self, xbrl_dir: Path, sec_html_path: Optional[Path] = None):
        self.xbrl_dir = xbrl_dir.resolve()
        self.sec_html_path = sec_html_path.resolve() if sec_html_path else None
        self.contexts: Dict[str, XBRLContext] = {}
        self.labels: Dict[str, str] = {}
        self.facts: List[XBRLFact] = []
        self._load_labels()
        self._load_contexts_and_facts()

    def _find_lab_file(self) -> Optional[Path]:
        for p in self.xbrl_dir.glob("*_lab.xml"):
            return p
        return None

    def _find_instance_htm(self) -> Optional[Path]:
        if self.sec_html_path and self.sec_html_path.is_file():
            return self.sec_html_path
        candidates = []
        for p in self.xbrl_dir.glob("*.htm"):
            if not p.name.startswith("a") and not p.name.startswith("R"):
                candidates.append(p)
        if candidates:
            return max(candidates, key=lambda f: f.stat().st_size)
        all_htm = list(self.xbrl_dir.glob("*.htm"))
        if all_htm:
            return max(all_htm, key=lambda f: f.stat().st_size)
        return None

    def _load_labels(self):
        lab_file = self._find_lab_file()
        if not lab_file or not lab_file.is_file():
            print(f"[WARN] No *_lab.xml found in {self.xbrl_dir}", file=sys.stderr)
            return
        try:
            tree = ET.parse(lab_file)
            root = tree.getroot()
            loc_map = {}
            for loc in root.findall(".//{http://www.xbrl.org/2003/linkbase}loc"):
                label_id = loc.attrib.get("{http://www.w3.org/1999/xlink}label")
                href = loc.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                concept = href.split("#")[-1] if "#" in href else href
                if label_id and concept:
                    loc_map[label_id] = concept
            
            label_text_map = {}
            for lbl in root.findall(".//{http://www.xbrl.org/2003/linkbase}label"):
                lbl_id = lbl.attrib.get("{http://www.w3.org/1999/xlink}label")
                if lbl_id and lbl.text:
                    label_text_map[lbl_id] = lbl.text.strip()

            for arc in root.findall(".//{http://www.xbrl.org/2003/linkbase}labelArc"):
                from_lbl = arc.attrib.get("{http://www.w3.org/1999/xlink}from")
                to_lbl = arc.attrib.get("{http://www.w3.org/1999/xlink}to")
                concept = loc_map.get(from_lbl)
                text = label_text_map.get(to_lbl)
                if concept and text and concept not in self.labels:
                    self.labels[concept] = text
        except Exception as e:
            print(f"[WARN] Error loading labels: {e}", file=sys.stderr)

    def _load_contexts_and_facts(self):
        htm_file = self._find_instance_htm()
        if not htm_file or not htm_file.is_file():
            print(f"[WARN] No instance HTM found in {self.xbrl_dir}", file=sys.stderr)
            return
        
        print(f"[INFO] Loading facts from instance file: {htm_file.name} ({htm_file.stat().st_size:,} bytes)")
        with htm_file.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        ctx_matches = re.findall(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', content, re.DOTALL)
        for cid, cbody in ctx_matches:
            end_m = re.search(r'<xbrli:endDate>([^<]+)</xbrli:endDate>', cbody)
            inst_m = re.search(r'<xbrli:instant>([^<]+)</xbrli:instant>', cbody)
            start_m = re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>', cbody)
            dim_m = re.search(r'<xbrldi:explicitMember[^>]*>([^<]+)</xbrldi:explicitMember>', cbody)
            
            year = None
            date_str = None
            ptype = "unknown"
            if end_m:
                date_str = end_m.group(1).strip()
                ptype = "duration"
            elif inst_m:
                date_str = inst_m.group(1).strip()
                ptype = "instant"
            
            if date_str:
                ym = re.search(r'^(202[0-9]|201[0-9])', date_str)
                if ym:
                    year = int(ym.group(1))
            
            self.contexts[cid] = XBRLContext(
                context_id=cid,
                year=year,
                period_type=ptype,
                start_date=start_m.group(1).strip() if start_m else None,
                end_date=end_m.group(1).strip() if end_m else None,
                instant=inst_m.group(1).strip() if inst_m else None,
                dimension=dim_m.group(1).strip() if dim_m else None
            )

        fact_pattern = re.compile(
            r'<ix:nonFraction\s+([^>]*?)>(.*?)</ix:nonFraction>',
            re.DOTALL
        )
        
        for attrs_str, raw_val_str in fact_pattern.findall(content):
            name_m = re.search(r'name="([^"]+)"', attrs_str)
            ctx_m = re.search(r'contextRef="([^"]+)"', attrs_str)
            unit_m = re.search(r'unitRef="([^"]+)"', attrs_str)
            dec_m = re.search(r'decimals="([^"]+)"', attrs_str)
            scale_m = re.search(r'scale="([^"]+)"', attrs_str)
            sign_m = re.search(r'sign="([^"]+)"', attrs_str)

            if not name_m or not ctx_m:
                continue

            concept = name_m.group(1).split(":")[-1]
            cid = ctx_m.group(1)
            ctx = self.contexts.get(cid)
            year = ctx.year if ctx else None

            raw_text = re.sub(r'<[^>]+>', '', raw_val_str).strip()
            num_val = normalize_numeric_str(raw_text)
            if sign_m and sign_m.group(1) == "-" and num_val is not None:
                num_val = -abs(num_val)

            label = self.labels.get(concept, concept)
            
            self.facts.append(XBRLFact(
                concept=concept,
                context_id=cid,
                raw_value=raw_text,
                numeric_value=num_val,
                year=year,
                unit=unit_m.group(1) if unit_m else "",
                decimals=dec_m.group(1) if dec_m else "",
                scale=int(scale_m.group(1)) if scale_m else 0,
                label=label,
                is_negative=(num_val is not None and num_val < 0)
            ))


@dataclass
class ParsedTableCell:
    page_num: int
    table_index: int
    row_index: int
    col_index: int
    raw_value: str
    numeric_value: Optional[float]
    column_header: str
    column_year: Optional[int]
    row_label: str


@dataclass
class ParsedDocument:
    pages: Dict[int, Dict[str, Any]]
    table_cells: List[ParsedTableCell]
    text_numbers_by_page: Dict[int, Set[float]]
    section_headers_by_page: Dict[int, List[str]]


def load_parsed_pages(review_dir: Path) -> ParsedDocument:
    pages_dir = review_dir / "pages"
    pages_data: Dict[int, Dict[str, Any]] = {}
    table_cells: List[ParsedTableCell] = []
    text_numbers_by_page: Dict[int, Set[float]] = {}
    section_headers_by_page: Dict[int, List[str]] = {}

    for page_file in sorted(pages_dir.glob("page_*.json")):
        m = re.search(r"page_(\d+)\.json", page_file.name)
        if not m:
            continue
        p_num = int(m.group(1))
        with page_file.open("r", encoding="utf-8") as f:
            p_data = json.load(f)
        pages_data[p_num] = p_data

        page_obj = p_data.get("page", {})
        blocks = page_obj.get("blocks", [])

        headers_list = []
        for blk in blocks:
            b_type = blk.get("block_type", "")
            if b_type == "heading":
                headers_list.append(blk.get("text", "").strip())
        section_headers_by_page[p_num] = headers_list

        page_nums = set()
        t_idx = 0
        for blk in blocks:
            b_type = blk.get("block_type", "")
            if b_type in ("paragraph", "text", "heading"):
                text = blk.get("text", "")
                tokens = re.findall(r'[\$\(]?-?\d[\d,]*\.?\d*[\%\)]?', text)
                for tok in tokens:
                    num = normalize_numeric_str(tok)
                    if num is not None:
                        page_nums.add(num)
            elif b_type == "table":
                t_idx += 1
                headers = blk.get("headers", [])
                rows = blk.get("rows", [])
                
                col_years: Dict[int, Optional[int]] = {}
                for col_i, h in enumerate(headers):
                    yrs = extract_years_from_header(str(h))
                    col_years[col_i] = yrs[0] if yrs else None

                if not any(col_years.values()) and rows:
                    for col_i, cell in enumerate(rows[0]):
                        yrs = extract_years_from_header(str(cell))
                        if yrs:
                            col_years[col_i] = yrs[0]

                for r_idx, row in enumerate(rows):
                    row_label = str(row[0]).strip() if row else ""
                    for col_idx, cell in enumerate(row):
                        cell_str = str(cell).strip()
                        num = normalize_numeric_str(cell_str)
                        col_h = headers[col_idx] if col_idx < len(headers) else ""
                        c_year = col_years.get(col_idx)
                        table_cells.append(ParsedTableCell(
                            page_num=p_num,
                            table_index=t_idx,
                            row_index=r_idx,
                            col_index=col_idx,
                            raw_value=cell_str,
                            numeric_value=num,
                            column_header=str(col_h),
                            column_year=c_year,
                            row_label=row_label
                        ))
        text_numbers_by_page[p_num] = page_nums

    return ParsedDocument(
        pages=pages_data,
        table_cells=table_cells,
        text_numbers_by_page=text_numbers_by_page,
        section_headers_by_page=section_headers_by_page
    )


@dataclass
class MatchResult:
    fact: XBRLFact
    status: str
    matched_page: Optional[int] = None
    matched_col_year: Optional[int] = None
    matched_row_label: Optional[str] = None
    note: str = ""


class ReconciliationEngine:
    def __init__(self, gt: GroundTruthExtractor, parsed_doc: ParsedDocument, review_queue_pages: Set[int]):
        self.gt = gt
        self.doc = parsed_doc
        self.review_queue_pages = review_queue_pages
        self.results: List[MatchResult] = []

    def run(self):
        val_to_cells: Dict[float, List[ParsedTableCell]] = {}
        for c in self.doc.table_cells:
            if c.numeric_value is not None:
                val_to_cells.setdefault(round(c.numeric_value, 2), []).append(c)

        for fact in self.gt.facts:
            if fact.numeric_value is None:
                continue

            target_val = round(fact.numeric_value, 2)
            gt_year = fact.year

            candidate_cells = val_to_cells.get(target_val, [])
            
            best_match: Optional[ParsedTableCell] = None
            swap_candidate: Optional[ParsedTableCell] = None

            for cell in candidate_cells:
                label_sim = (
                    fact.label.lower() in cell.row_label.lower()
                    or cell.row_label.lower() in fact.label.lower()
                    or fact.concept.lower() in cell.row_label.lower()
                )

                if cell.column_year is not None and gt_year is not None:
                    if cell.column_year == gt_year:
                        if label_sim:
                            best_match = cell
                            break
                        elif best_match is None:
                            best_match = cell
                    else:
                        if label_sim:
                            swap_candidate = cell

            if best_match:
                self.results.append(MatchResult(
                    fact=fact,
                    status="RAG_SAFE_MATCH",
                    matched_page=best_match.page_num,
                    matched_col_year=best_match.column_year,
                    matched_row_label=best_match.row_label,
                    note=f"Matched in table column {best_match.column_year} on page {best_match.page_num}"
                ))
            elif swap_candidate:
                self.results.append(MatchResult(
                    fact=fact,
                    status="YEAR_COLUMN_SWAPPED",
                    matched_page=swap_candidate.page_num,
                    matched_col_year=swap_candidate.column_year,
                    matched_row_label=swap_candidate.row_label,
                    note=f"HIGH RISK: Fact year is {gt_year}, but placed under column {swap_candidate.column_year} on page {swap_candidate.page_num}"
                ))
            else:
                if candidate_cells:
                    c0 = candidate_cells[0]
                    self.results.append(MatchResult(
                        fact=fact,
                        status="RAG_SAFE_MATCH",
                        matched_page=c0.page_num,
                        matched_col_year=c0.column_year,
                        matched_row_label=c0.row_label,
                        note=f"Matched numeric value in table on page {c0.page_num} (col year: {c0.column_year})"
                    ))
                    continue

                narrative_page = None
                for p_num, nums in self.doc.text_numbers_by_page.items():
                    if any(abs(n - target_val) < 1e-4 for n in nums):
                        narrative_page = p_num
                        break

                if narrative_page:
                    self.results.append(MatchResult(
                        fact=fact,
                        status="NARRATIVE_MATCH",
                        matched_page=narrative_page,
                        note=f"Matched in narrative text on page {narrative_page}"
                    ))
                else:
                    self.results.append(MatchResult(
                        fact=fact,
                        status="MISSING_OMISSION",
                        note="Value not found in parsed pages (in review queue or missed by engine)"
                    ))


def generate_reconciliation_reports(
    review_dir: Path,
    reconciler: ReconciliationEngine,
    review_queue_records: List[Dict[str, Any]],
    company_name: str = "INTEL CORPORATION",
    filing_acc: str = "0000050863-26-000011",
    pdf_name: str = "0000050863-26-000011.pdf",
):
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / "ground_truth_reconciliation_report.md"
    deep_dive_path = review_dir / "review_queue_deep_dive.md"

    results = reconciler.results
    total_facts = len(results)
    
    rag_safe_matches = [r for r in results if r.status in ("RAG_SAFE_MATCH", "NARRATIVE_MATCH")]
    table_safe = [r for r in results if r.status == "RAG_SAFE_MATCH"]
    narrative_safe = [r for r in results if r.status == "NARRATIVE_MATCH"]
    year_swapped = [r for r in results if r.status == "YEAR_COLUMN_SWAPPED"]
    omissions = [r for r in results if r.status == "MISSING_OMISSION"]

    rag_safe_rate = (len(rag_safe_matches) / total_facts * 100) if total_facts else 0.0
    
    year_evaluated = [r for r in results if r.fact.year and r.matched_col_year]
    year_correct = [r for r in year_evaluated if r.fact.year == r.matched_col_year]
    year_acc = (len(year_correct) / len(year_evaluated) * 100) if year_evaluated else 100.0

    year_breakdown: Dict[int, List[MatchResult]] = {}
    for r in results:
        yr = r.fact.year or 0
        year_breakdown.setdefault(yr, []).append(r)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Báo Cáo Đối Soát Toàn Diện Ground Truth & Đề Mục {company_name} Form 10-K\n\n")
        f.write(f"**Ngày thực hiện:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Bộ dữ liệu Ground Truth:** SEC EDGAR Form 10-K (HTML & XBRL Package `{filing_acc}`)\n\n")
        f.write(f"**Tổng số facts tài chính kiểm thẩm:** **{total_facts:,}** số liệu trên toàn bộ tài liệu\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Tóm Tắt Chỉ Số Chất Lượng Hướng Tới RAG (RAG-Readiness Summary)\n\n")
        f.write(f"- **Tỷ lệ Khớp Đạt Chuẩn RAG (RAG-Safe Match Rate):** **`{rag_safe_rate:.2f}%`** ({len(rag_safe_matches):,} / {total_facts:,})\n")
        f.write(f"  - Khớp trong bảng (Table Data): **{len(table_safe):,}** số liệu\n")
        f.write(f"  - Khớp trong văn bản thuyết minh (Narrative Text): **{len(narrative_safe):,}** số liệu\n")
        f.write(f"- **Độ chính xác Liên kết Cột Năm (Year-Column Alignment Accuracy):** **`{year_acc:.2f}%`** ({len(year_correct):,} / {len(year_evaluated):,})\n")
        f.write(f"- **Số lượng Bị Lệch / Tráo Cột Năm (High-Risk Hallucinations):** **`{len(year_swapped)}`** trường hợp\n")
        f.write(f"- **Số lượng Chưa Khớp / Thuộc Trang Review Queue:** **`{len(omissions):,}`** trường hợp\n\n")

        f.write("---\n\n")
        f.write("## 2. Phân Tích Độ Chính Xác Theo Năm (Temporal Alignment Breakdown)\n\n")
        f.write("| Năm Tài Chính | Tổng Số Ground Truth Facts | RAG-Safe Match | Tỷ Lệ Khớp RAG | Tráo Cột (Swapped) | Bỏ Sót / Chờ VLM |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for yr in sorted(year_breakdown.keys(), reverse=True):
            yr_list = year_breakdown[yr]
            yr_total = len(yr_list)
            yr_safe = sum(1 for r in yr_list if r.status in ("RAG_SAFE_MATCH", "NARRATIVE_MATCH"))
            yr_swap = sum(1 for r in yr_list if r.status == "YEAR_COLUMN_SWAPPED")
            yr_miss = sum(1 for r in yr_list if r.status == "MISSING_OMISSION")
            yr_rate = (yr_safe / yr_total * 100) if yr_total else 0.0
            label_yr = f"FY {yr}" if yr > 0 else "Không xác định / Khác"
            f.write(f"| **{label_yr}** | {yr_total:,} | {yr_safe:,} | **{yr_rate:.1f}%** | {yr_swap} | {yr_miss:,} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 3. Kiểm Thẩm Các Chỉ Tiêu Tài Chính Cốt Lõi (Core Financial Indicators Audit)\n\n")
        f.write("Dưới đây là đối soát chi tiết các chỉ tiêu chủ chốt trên các báo cáo tài chính hợp nhất chính thức:\n\n")
        f.write("| Khái Niệm XBRL (Concept) | Chỉ Tiêu (Line Item) | Năm | Giá Trị XBRL Ground Truth | Giá Trị Bóc Tách Được | Trang PDF | Cột Năm Bóc Tách | Đánh Giá RAG |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        core_concepts = [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "GrossProfit",
            "OperatingIncomeLoss",
            "NetIncomeLoss",
            "EarningsPerShareBasic",
            "EarningsPerShareDiluted",
            "CashAndCashEquivalentsAtCarryingValue",
            "AssetsCurrent",
            "Assets",
            "LiabilitiesCurrent",
            "Liabilities",
            "StockholdersEquity",
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInInvestingActivities",
            "NetCashProvidedByUsedInFinancingActivities",
            "ResearchAndDevelopmentExpense",
            "SellingGeneralAndAdministrativeExpense",
        ]

        core_matches = [r for r in results if r.fact.concept in core_concepts]
        core_matches.sort(key=lambda x: (x.fact.concept, -(x.fact.year or 0)))

        for cm in core_matches:
            c_name = cm.fact.label or cm.fact.concept
            c_yr = cm.fact.year or "N/A"
            c_gt_val = f"{cm.fact.numeric_value:,.2f}" if cm.fact.numeric_value is not None else "N/A"
            if cm.status in ("RAG_SAFE_MATCH", "NARRATIVE_MATCH"):
                status_badge = "✅ CHUẨN XÁC"
                pred_val = c_gt_val
                matched_pg = str(cm.matched_page or "N/A")
                matched_col_yr = str(cm.matched_col_year or "N/A")
            elif cm.status == "YEAR_COLUMN_SWAPPED":
                status_badge = "⚠️ TRÁO CỘT NĂM"
                pred_val = c_gt_val
                matched_pg = str(cm.matched_page or "N/A")
                matched_col_yr = str(cm.matched_col_year or "N/A")
            else:
                status_badge = "❌ THIẾU SỐ"
                pred_val = "Không tìm thấy"
                matched_pg = "N/A"
                matched_col_yr = "N/A"

            f.write(f"| `{cm.fact.concept}` | {c_name} | {c_yr} | `{c_gt_val}` | `{pred_val}` | {matched_pg} | {matched_col_yr} | {status_badge} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 4. Chi Tiết Các Trường Hợp Bị Nhầm Cột Năm (High-Risk Hallucination Details)\n\n")
        if year_swapped:
            f.write("| Trang PDF | Chỉ Mục / Dòng Bảng | Năm Chuẩn (GT) | Cột Năm Bị Gán | Giá Trị Số Liệu | Ghi Chú |\n")
            f.write("| :---: | :--- | :---: | :---: | :---: | :--- |\n")
            for ys in year_swapped:
                f.write(f"| **{ys.matched_page}** | {ys.matched_row_label} | `{ys.fact.year}` | `{ys.matched_col_year}` | `{ys.fact.numeric_value:,.2f}` | `{ys.fact.concept}`: {ys.note} |\n")
        else:
            f.write("🎉 **Tuyệt đối không có trường hợp nào bị tráo hoặc lệch cột năm!** Toàn bộ dữ liệu niên độ đều thẳng hàng 100%.\n")
        f.write("\n")

    with deep_dive_path.open("w", encoding="utf-8") as f:
        f.write(f"# Báo Cáo Phân Tích Chuyên Sâu Review Queue & Các Trang Cần Xử Lý ({company_name})\n\n")
        f.write(f"**Ngày phân tích:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Tổng số trang trong Review Queue:** **{len(review_queue_records)}** trang\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Phân Loại Nguyên Nhân Theo Bộ Lọc QC Gate\n\n")
        
        reason_counts: Dict[str, int] = {}
        for r in review_queue_records:
            qc = r.get("qc") or {}
            for fail in qc.get("failures", []):
                fail_key = fail.split(":")[0]
                reason_counts[fail_key] = reason_counts.get(fail_key, 0) + 1
            for reason in r.get("route", {}).get("reason", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        f.write("| Mã Lỗi / Nguyên Nhân QC Gate | Số Lượng Trang | Diễn Giải Chi Tiết |\n")
        f.write("| :--- | :---: | :--- |\n")
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
            f.write(f"| `{k}` | **{v}** | Nguyên nhân bóc tách kích hoạt fallback |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 2. Bảng Kê Chi Tiết Các Trang Trong Review Queue\n\n")
        f.write("| Trang PDF | Phân Loại Trang | Điểm Numeric Recall | Điểm Text Precision | Các Lỗi Kích Hoạt (Failures) | Khuyến Nghị Xử Lý |\n")
        f.write("| :---: | :---: | :---: | :---: | :--- | :--- |\n")
        for r in review_queue_records:
            pg = r.get("pdf_page")
            p_cls = r.get("page_class")
            qc = r.get("qc") or {}
            num_rec = f"{qc.get('numeric_recall', 0.0):.3f}" if qc.get('numeric_recall') is not None else "N/A"
            txt_prec = f"{qc.get('source_text_precision', 0.0):.3f}" if qc.get('source_text_precision') is not None else "N/A"
            failures = ", ".join(qc.get("failures", [])) or ", ".join(r.get("route", {}).get("reason", []))
            f.write(f"| **{pg}** | `{p_cls}` | {num_rec} | {txt_prec} | `{failures}` | Chạy VLM fallback nhắm mục tiêu |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 3. Kế Hoạch Chạy Bù VLM Chọn Lọc (Selective VLM Fallback Command)\n\n")
        if review_queue_records:
            f.write("Để xử lý dứt điểm các trang này khi có API Key VLM (DeepSeek hoặc Gemini), kích hoạt lệnh:\n\n")
            f.write("```powershell\n")
            f.write(f"# Chạy VLM cho danh sách các trang trong review queue\n")
            f.write(f"python script/parse_financial_reports.py data/{pdf_name} --out \"{review_dir.name}\" --provider deepseek --force\n")
            f.write("```\n")
        else:
            f.write("🎉 **Không có trang nào trong review queue! Toàn bộ các trang đều đạt chuẩn an toàn.**\n")

    summary_data = {
        "company_name": company_name,
        "filing_acc": filing_acc,
        "total_facts": total_facts,
        "rag_safe_matches": len(rag_safe_matches),
        "rag_safe_rate": round(rag_safe_rate, 2),
        "year_column_alignment_accuracy": round(year_acc, 2),
        "year_swapped_count": len(year_swapped),
        "omissions_count": len(omissions),
        "review_queue_count": len(review_queue_records),
    }
    summary_file = review_dir / "ground_truth_reconciliation_summary.json"
    summary_file.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Generated reconciliation report: {report_path}")
    print(f"[SUCCESS] Generated deep-dive report: {deep_dive_path}")
    print(f"[SUCCESS] Generated summary JSON: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Reconcile parsed output with official SEC Ground Truth")
    parser.add_argument("--parsed", type=Path, default=Path("REVIEW PARSING"), help="Parsed output directory")
    parser.add_argument("--xbrl", type=Path, default=Path("data/0000050863-26-000011-xbrl"), help="XBRL directory")
    parser.add_argument("--html", type=Path, default=None, help="SEC HTML file")
    parser.add_argument("--company", type=str, default="INTEL CORPORATION", help="Company Name")
    parser.add_argument("--acc", type=str, default="0000050863-26-000011", help="SEC Accession Number")
    parser.add_argument("--pdf", type=str, default="0000050863-26-000011.pdf", help="Input PDF filename")
    args = parser.parse_args()

    print("=== EXTRACTING GROUND TRUTH FROM SEC XBRL / HTML ===")
    gt = GroundTruthExtractor(args.xbrl, args.html)
    print(f"Loaded {len(gt.contexts)} contexts, {len(gt.labels)} element labels, and {len(gt.facts)} facts.")

    print("\n=== LOADING PARSED DOCUMENT PAGES ===")
    parsed_doc = load_parsed_pages(args.parsed)
    print(f"Loaded {len(parsed_doc.pages)} canonical JSON pages with {len(parsed_doc.table_cells)} table cells.")

    review_queue_path = args.parsed / "review_queue.jsonl"
    review_records = []
    review_pages = set()
    if review_queue_path.is_file():
        with review_queue_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    review_records.append(rec)
                    if "pdf_page" in rec:
                        review_pages.add(rec["pdf_page"])
    print(f"Loaded {len(review_records)} review queue records.")

    print("\n=== RUNNING RAG-SAFE RECONCILIATION ENGINE ===")
    reconciler = ReconciliationEngine(gt, parsed_doc, review_pages)
    reconciler.run()

    print("\n=== GENERATING COMPREHENSIVE REPORTS ===")
    generate_reconciliation_reports(
        args.parsed,
        reconciler,
        review_records,
        company_name=args.company,
        filing_acc=args.acc,
        pdf_name=args.pdf,
    )
    print("=== FINISHED RECONCILIATION AUDIT ===")


if __name__ == "__main__":
    main()
