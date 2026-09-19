"""Post-parse gates used for automatic fallback, never for automatic correction."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from .config import ParserConfig
from .markdown_utils import (
    NUMBER_PATTERN,
    extract_markdown_tables,
    multiset_precision,
    multiset_recall,
    number_multiset,
    token_multiset,
)
from .models import EngineName, PageProfile, QCResult, QCStatus

DATE_PATTERNS = [
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b", re.I),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
]
RANGE_PATTERN = re.compile(r"^\s*[$₫€£¥]?\s*\(?[\d.,]+%?\)?\s*(?:[-–—]|to)\s*[$₫€£¥]?\s*\(?[\d.,]+%?\)?\s*$", re.I)
SEC_CODE_PATTERN = re.compile(r"^\s*\d{3,}[-–—]\d+\s*$")


def _is_suspicious_merged_numeric_cell(cell: str, cell_index: int, row: List[str]) -> bool:
    """Detect if a numeric data column merged multiple numbers from adjacent columns.

    Guards ensure:
    - Column 0 is never flagged (it is row labels, section headers, dates, item titles).
    - Descriptive cells (narrative sentences with >= 4 alphabetic words) are excluded.
    - Dates, numeric ranges (e.g. 2025-2062, 0.03%-5.75%), and SEC filing codes are excluded.

    Strict mode: only flag when there is positive evidence of an empty sibling cell in the same row.
    Trade-off: accepts potential false negatives to strictly avoid false positives on SEC IDs / ranges.
    """
    if cell_index == 0:
        return False

    c = cell.strip()
    if not c:
        return False

    # Narrative/explanatory columns (e.g. Audit report, Exhibit descriptions)
    words = [w for w in c.split() if any(ch.isalpha() for ch in w)]
    if len(words) >= 4:
        return False

    for dp in DATE_PATTERNS:
        if dp.search(c):
            return False

    if RANGE_PATTERN.match(c) or SEC_CODE_PATTERN.match(c):
        return False

    matches = NUMBER_PATTERN.findall(c)
    if len(matches) <= 1:
        return False

    # Bug 10 fix: use row as proof of merged column — 0-based scan across all sibling cells.
    has_empty_sibling = any(
        idx != cell_index and not r.strip()
        for idx, r in enumerate(row)
    )
    return has_empty_sibling


def _table_shape(markdown: str) -> Tuple[bool, int, List[str]]:
    tables = extract_markdown_tables(markdown)
    warnings: List[str] = []
    for table_index, table in enumerate(tables, 1):
        width = len(table["headers"])
        if width < 2:
            warnings.append(f"table_{table_index}_has_less_than_two_columns")
        for row_index, row in enumerate(table["rows"], 1):
            if len(row) != width:
                warnings.append(
                    f"table_{table_index}_row_{row_index}_has_{len(row)}_cells_expected_{width}"
                )

            for cell_index, cell in enumerate(row):
                if _is_suspicious_merged_numeric_cell(cell, cell_index, row):
                    warnings.append(
                        f"table_{table_index}_row_{row_index}_cell_{cell_index + 1}_possible_merged_numeric_cell"
                    )
    return not warnings, len(tables), warnings


VERB_PATTERN = re.compile(
    r"\b(is|are|was|were|has|have|had|will|would|can|could|may|might|"
    r"do|does|did|be|been|being|"
    r"reports?|reported|reporting|increases?|increased|increasing|"
    r"decreases?|decreased|decreasing|grew|grow|growing|"
    r"updates?|updated|updating|continues?|continued|continuing|"
    r"requires?|required|requiring|limits?|limited|limiting|"
    r"specifies?|specified|specifying|imposes?|imposed|imposing|"
    r"maintains?|maintained|maintaining|operates?|operated|operating|"
    r"includes?|included|including|provides?|provided|providing|"
    r"consists?|consisted|consisting|agrees?|agreed|agreeing|"
    r"là|được|có|đã|sẽ|tăng|giảm|báo cáo|cập nhật|tiếp tục|yêu cầu|quy định|áp dụng|bao gồm|cung cấp)\b",
    re.IGNORECASE,
)

NUMBER_ROW_PATTERN = re.compile(
    r"(?:[-−–—]?[$₫€£¥]?[ \t]*\d[\d,.]*[%]?\s+){2,}[-−–—]?[$₫€£¥]?[ \t]*\d[\d,.]*[%]?"
)


def _has_sentence_structure(text: str) -> bool:
    """>= 3 complete sentences or narrative bullets with grammatical verbs."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullet_lines = sum(
        1
        for ln in lines
        if (re.match(r"^[-*+]\s+", ln) or re.match(r"^\d+\.\s+", ln))
        and len(ln.split()) >= 6
        and VERB_PATTERN.search(ln)
    )
    if bullet_lines >= 3:
        return True

    # Splitting avoids decimal numbers like 0.500% or 1.35%
    sentences = [
        s.strip() for s in re.split(r"[!?]|\.(?!\d)(?:\s+|$)", text) if s.strip()
    ]
    complete = sum(
        1 for s in sentences if VERB_PATTERN.search(s) and len(s.split()) >= 8
    )
    return complete >= 3


def _looks_like_flattened_table(text: str) -> bool:
    """Fail-safe detector: Return True if text shows structural signals of a flattened table.

    Signals:
    1. Short numeric line ratio: >= 50% of non-empty lines are short (<= 8 words)
       and contain numbers (typical table rows broken into lines).
    2. Spaced numbers in row: >= 3 lines contain multiple space-separated numbers
       (typical table data rows).
    3. Lack of sentence grammar structure: does not contain >= 3 complete sentences/bullets with verbs.

    Fail-safe rule: If ANY signal is True -> True (FAIL).
    Only when ALL signals are False -> False (Narrative misdetect).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True

    # Signal 1: Short numeric line ratio
    short_numeric_lines = sum(
        1 for ln in lines if len(ln.split()) <= 8 and re.search(r"\d", ln)
    )
    if (short_numeric_lines / max(1, len(lines))) >= 0.5:
        return True

    # Signal 2: Lines with multiple space-separated numbers
    number_rows = sum(1 for ln in lines if NUMBER_ROW_PATTERN.search(ln))
    if number_rows >= 3:
        return True

    # Signal 3: Lacks grammatical sentence structure
    if not _has_sentence_structure(text):
        return True

    return False


def evaluate_output(
    markdown: str,
    profile: PageProfile,
    engine: EngineName,
    config: ParserConfig,
    prev_page_markdown: Optional[str] = None,
    next_page_markdown: Optional[str] = None,
) -> QCResult:
    """Evaluate parser output conservatively.

    Native PDF text is a zero-cost source oracle for Docling.  It does not prove
    table associations, so a passing result means "safe enough to continue",
    not that the page is ground-truth correct.
    """
    value = markdown.strip()
    if not value:
        return QCResult(
            status=QCStatus.FAIL,
            source_text_recall=0.0 if profile.raw_text else None,
            numeric_recall=0.0 if profile.raw_text else None,
            numeric_precision=0.0 if profile.raw_text else None,
            table_shape_pass=False,
            table_count=0,
            failures=["empty_parser_output"],
        )

    shape_pass, table_count, structure_findings = _table_shape(value)
    source_is_reliable = profile.page_class.value == "native_text" and bool(profile.raw_text)
    source_text_recall: Optional[float] = None
    numeric_recall: Optional[float] = None
    numeric_precision: Optional[float] = None
    failures: List[str] = []
    warnings: List[str] = list(structure_findings)
    source_numbers: Counter = number_multiset(profile.raw_text) if profile.raw_text else Counter()

    if source_is_reliable:
        source_tokens = token_multiset(profile.raw_text)
        predicted_tokens = token_multiset(value)
        source_text_recall = multiset_recall(source_tokens, predicted_tokens)
        predicted_numbers = number_multiset(value)
        numeric_recall = multiset_recall(source_numbers, predicted_numbers)
        numeric_precision = multiset_precision(source_numbers, predicted_numbers)

        # Cross-page boundary recall resolution (text and numbers):
        # A single paragraph spanning page breaks may be attached to the previous page
        # or next page by a document-level parser (like Docling).
        if source_tokens and source_text_recall < config.min_docling_text_recall:
            missing_tokens = source_tokens - predicted_tokens
            resolved_tokens = Counter()
            if prev_page_markdown:
                resolved_tokens |= (missing_tokens & token_multiset(prev_page_markdown))
            if next_page_markdown:
                resolved_tokens |= (missing_tokens & token_multiset(next_page_markdown))
            if resolved_tokens:
                adj_tokens = predicted_tokens + resolved_tokens
                adj_text_rec = multiset_recall(source_tokens, adj_tokens)
                if adj_text_rec >= config.min_docling_text_recall:
                    warnings.append(f"cross_page_boundary_resolved_text_recall:{adj_text_rec:.3f}")
                    source_text_recall = adj_text_rec

        if source_numbers and numeric_recall < config.min_docling_numeric_recall:
            missing = source_numbers - predicted_numbers
            resolved = Counter()
            if prev_page_markdown:
                resolved |= (missing & number_multiset(prev_page_markdown))
            if next_page_markdown:
                resolved |= (missing & number_multiset(next_page_markdown))
            if resolved:
                adj_predicted_numbers = predicted_numbers + resolved
                adj_recall = multiset_recall(source_numbers, adj_predicted_numbers)
                if adj_recall >= config.min_docling_numeric_recall:
                    warnings.append(f"cross_page_boundary_resolved_numeric_recall:{adj_recall:.3f}")
                    numeric_recall = adj_recall

        # Precision adjustment for repeated table year headers & bounded cross-page overflow:
        # Repeating column fiscal year headers (e.g. 2025, 2024) across sub-tables
        # is structurally beneficial and must not fail the precision gate.
        if (
            predicted_numbers
            and numeric_precision < config.min_docling_numeric_precision
            and numeric_recall >= config.min_recall_for_adjustment
        ):
            extra = predicted_numbers - source_numbers
            common = source_numbers & predicted_numbers

            # Bug 4 fix: only exempt years actually appearing in Markdown table HEADERS,
            # never data cells or row labels.
            header_years: set = set()
            for table in extract_markdown_tables(value):
                header_text = " ".join(table["headers"])
                for num in number_multiset(header_text):
                    if len(num) == 4 and num.isdigit() and 1900 <= int(num) <= 2099:
                        header_years.add(num)

            # 1. Header-year exemption (uncapped — structurally valid)
            header_year_exempt = sum(
                count for num, count in extra.items() if num in header_years
            )

            # 2. Overflow exemption (capped at 50% — benefit of the doubt)
            non_year_extra = extra - Counter(
                {n: c for n, c in extra.items() if n in header_years}
            )
            adjacent_numbers = Counter()
            if prev_page_markdown:
                adjacent_numbers |= number_multiset(prev_page_markdown)
            if next_page_markdown:
                adjacent_numbers |= number_multiset(next_page_markdown)

            overflow_candidates = sum(
                (non_year_extra & adjacent_numbers).values()
            )
            overflow_cap = max(1, int(0.5 * sum(non_year_extra.values())))
            overflow_exempt = min(overflow_candidates, overflow_cap)

            exempt_count = header_year_exempt + overflow_exempt

            adj_pred_total = max(1, sum(predicted_numbers.values()) - exempt_count)
            adj_precision = sum(common.values()) / adj_pred_total
            if adj_precision >= config.min_docling_numeric_precision:
                warnings.append(
                    f"header_year_repetition_adjusted_precision:{adj_precision:.3f}"
                )
                numeric_precision = adj_precision
            else:
                warnings.append(
                    f"exemption_insufficient:raw={numeric_precision:.3f},"
                    f"adj={adj_precision:.3f},"
                    f"header_year={header_year_exempt},"
                    f"overflow={overflow_exempt}"
                )

        # The gates apply to either engine whenever the native PDF text is a
        # reliable oracle. A VLM fallback must not bypass a known numeric loss.
        is_narrative_page = (
            source_text_recall is not None
            and source_text_recall >= config.min_narrative_text_recall
            and (
                table_count == 0
                or (table_count <= 1 and sum(source_numbers.values()) <= config.max_narrative_numbers)
            )
        )

        if source_text_recall < config.min_docling_text_recall:
            failures.append(f"source_text_recall_below_gate:{source_text_recall:.3f}")
        if source_numbers and numeric_recall < config.min_docling_numeric_recall:
            if is_narrative_page:
                warnings.append(f"narrative_minor_numeric_delta:recall={numeric_recall:.3f}")
            else:
                failures.append(f"numeric_recall_below_gate:{numeric_recall:.3f}")
        if predicted_numbers and numeric_precision < config.min_docling_numeric_precision:
            if is_narrative_page:
                warnings.append(f"narrative_minor_numeric_delta:precision={numeric_precision:.3f}")
            else:
                failures.append(f"numeric_precision_below_gate:{numeric_precision:.3f}")
    else:
        warnings.append("no_reliable_native_text_oracle_for_fidelity_check")

    if not shape_pass:
        # A malformed Markdown table is unsafe for Structure Chunking regardless
        # of which parser generated it.
        failures.extend(item for item in structure_findings if item not in failures)
    if profile.likely_tabular and table_count == 0:
        # Native geometry suggests tabular layout. Missing table in output
        # = lost structure = FAIL, unless fail-safe structural signals confirm genuine narrative text.
        if engine == EngineName.TATR:
            failures.append("table_engine_failed_to_produce_markdown_table")
        elif _looks_like_flattened_table(value):
            failures.append(
                "native_geometry_suggests_a_table_but_output_has_no_markdown_table"
            )
        else:
            warnings.append(
                "native_geometry_suggests_a_table_but_output_has_no_markdown_table"
            )

    if failures:
        status = QCStatus.FAIL
    elif warnings:
        status = QCStatus.WARNING
    else:
        status = QCStatus.PASS

    return QCResult(
        status=status,
        source_text_recall=round(source_text_recall, 4) if source_text_recall is not None else None,
        numeric_recall=round(numeric_recall, 4) if numeric_recall is not None else None,
        numeric_precision=round(numeric_precision, 4) if numeric_precision is not None else None,
        table_shape_pass=shape_pass,
        table_count=table_count,
        warnings=warnings,
        failures=failures,
    )
