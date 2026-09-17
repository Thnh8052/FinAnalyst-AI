"""Post-parse gates used for automatic fallback, never for automatic correction."""

from __future__ import annotations

import re
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

            # Two independent numeric groups in a cell are a strong signature
            # of merged cells, especially when the following cell is blank.
            for cell_index, cell in enumerate(row):
                matches = NUMBER_PATTERN.findall(cell)
                next_is_blank = cell_index + 1 < len(row) and not row[cell_index + 1].strip()
                # The first column is usually a description and can legitimately
                # contain dates, share counts and footnote numbers. Numeric data
                # columns should contain one financial value per cell.
                suspicious_numeric_cell = cell_index > 0 and len(matches) > 1
                if suspicious_numeric_cell or (len(matches) > 1 and next_is_blank):
                    warnings.append(
                        f"table_{table_index}_row_{row_index}_cell_{cell_index + 1}_possible_merged_numeric_cell"
                    )
    return not warnings, len(tables), warnings


def evaluate_output(
    markdown: str,
    profile: PageProfile,
    engine: EngineName,
    config: ParserConfig,
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

    if source_is_reliable:
        source_tokens = token_multiset(profile.raw_text)
        predicted_tokens = token_multiset(value)
        source_text_recall = multiset_recall(source_tokens, predicted_tokens)
        source_numbers = number_multiset(profile.raw_text)
        predicted_numbers = number_multiset(value)
        numeric_recall = multiset_recall(source_numbers, predicted_numbers)
        numeric_precision = multiset_precision(source_numbers, predicted_numbers)

        # The gates apply to either engine whenever the native PDF text is a
        # reliable oracle. A VLM fallback must not bypass a known numeric loss.
        if source_text_recall < config.min_docling_text_recall:
            failures.append(f"source_text_recall_below_gate:{source_text_recall:.3f}")
        if numeric_recall < config.min_docling_numeric_recall:
            failures.append(f"numeric_recall_below_gate:{numeric_recall:.3f}")
        if numeric_precision < config.min_docling_numeric_precision:
            failures.append(f"numeric_precision_below_gate:{numeric_precision:.3f}")
    else:
        warnings.append("no_reliable_native_text_oracle_for_fidelity_check")

    if not shape_pass:
        # A malformed Markdown table is unsafe for Structure Chunking regardless
        # of which parser generated it.
        failures.extend(item for item in structure_findings if item not in failures)
    if profile.likely_tabular and table_count == 0:
        failures.append("native_geometry_suggests_a_table_but_output_has_no_markdown_table")

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
