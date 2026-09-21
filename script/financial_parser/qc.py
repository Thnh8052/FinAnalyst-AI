"""Post-parse gates used for automatic fallback, never for automatic correction."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, NamedTuple, Optional, Tuple

from .config import ParserConfig
from .markdown_utils import (
    NUMBER_PATTERN,
    classify_number,
    extract_numbers_with_context,
    extract_markdown_tables,
    is_resolvable_token,
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

# Digit-boundary year patterns: match FY2025, Q1 2025, standalone 2025.
# Word boundary (\b) fails between Y and 2 in "FY2025" because both are \w.
YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
YEAR_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?:FY)?(?:19|20)\d{2}\s*[-\u2013\u2014]\s*(?:FY)?(?:19|20)\d{2}(?!\d)"
)


class TableShapeResult(NamedTuple):
    """Return type for _table_shape: separates hard failures from soft warnings."""
    pass_: bool               # True if no hard findings
    table_count: int
    hard_findings: List[str]  # structure defects → FAIL
    soft_findings: List[str]  # heuristic warnings → WARNING


def _adjacent_dedup(years: List[int]) -> List[int]:
    """Remove adjacent duplicate years.  [2025,2025,2024,2024] → [2025,2024].

    Empty input → empty output (guard IndexError).
    Non-adjacent duplicates preserved: [2025,2024,2025] → [2025,2024,2025].
    """
    if not years:
        return []
    return [years[0]] + [
        y for i, y in enumerate(years[1:], 1) if y != years[i - 1]
    ]


def _extract_header_years(headers: List[str]) -> List[int]:
    """Extract standalone years from table headers.

    - Digit boundary (not word boundary) → matches FY2025, Q1 2025
    - Skip range cells ("2024-2025", "FY2024–FY2025") → prevents false zigzag
    - Only years in [1900, 2099]
    """
    years: List[int] = []
    for h in headers:
        if YEAR_RANGE_PATTERN.search(h):
            continue
        for m in YEAR_PATTERN.finditer(h):
            y = int(m.group())
            if 1900 <= y <= 2099:
                years.append(y)
    return years


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


def _table_shape(markdown: str) -> TableShapeResult:
    """Check structure and shape of Markdown tables.

    Returns TableShapeResult with hard findings (→ FAIL) and soft findings (→ WARNING).

    Year monotonic check:
        LIMITATION: Chỉ bắt được zigzag pattern.
        Không bắt được reversal đơn giản ([2025, 2024] → [2024, 2025])
        vì cả 2 đều monotonic.  Full fix cần PDF coords (Bug 8 full).
    """
    tables = extract_markdown_tables(markdown)
    hard: List[str] = []
    soft: List[str] = []

    for table_index, table in enumerate(tables, 1):
        width = len(table["headers"])

        # ── HARD: table width ──
        if width < 2:
            hard.append(f"table_{table_index}_has_less_than_two_columns")

        # ── SOFT: year order (after adjacent dedup, with range cell skip) ──
        header_years = _extract_header_years(table["headers"])
        deduped = _adjacent_dedup(header_years)
        if len(deduped) >= 2:
            is_inc = all(deduped[i] < deduped[i + 1] for i in range(len(deduped) - 1))
            is_dec = all(deduped[i] > deduped[i + 1] for i in range(len(deduped) - 1))
            if not (is_inc or is_dec):
                soft.append(f"table_{table_index}_header_year_order_non_monotonic")

        # ── Per-row checks ──
        numeric_cells_per_col = [0] * width
        text_cells_per_col = [0] * width

        for row_index, row in enumerate(table["rows"], 1):
            # HARD: cell count mismatch
            if len(row) != width:
                hard.append(
                    f"table_{table_index}_row_{row_index}_has_{len(row)}_cells_expected_{width}"
                )

            for cell_index, cell in enumerate(row):
                # HARD: merged numeric cell
                if _is_suspicious_merged_numeric_cell(cell, cell_index, row):
                    hard.append(
                        f"table_{table_index}_row_{row_index}_cell_{cell_index + 1}_possible_merged_numeric_cell"
                    )

                # Track column types for inconsistency check
                if cell_index < width:
                    c_clean = cell.strip()
                    if c_clean:
                        has_num = bool(NUMBER_PATTERN.search(c_clean))
                        words = [w for w in c_clean.split() if any(ch.isalpha() for ch in w)]
                        if has_num and len(words) <= 2:
                            numeric_cells_per_col[cell_index] += 1
                        elif len(words) >= 4:
                            text_cells_per_col[cell_index] += 1

        # ── SOFT: column data type inconsistency (heuristic) ──
        total_rows = len(table["rows"])
        if total_rows >= 4:
            for col_idx in range(1, width):
                if numeric_cells_per_col[col_idx] >= max(3, int(0.6 * total_rows)):
                    if text_cells_per_col[col_idx] >= 2:
                        soft.append(
                            f"table_{table_index}_column_{col_idx + 1}_inconsistent_data_types"
                        )

    # Cross-table year direction check: REMOVED.
    # Without past/future context, comparing historical (desc) vs maturity (asc)
    # tables on the same page produces false positives.

    return TableShapeResult(
        pass_=not hard,
        table_count=len(tables),
        hard_findings=hard,
        soft_findings=soft,
    )


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
            source_text_precision=0.0 if profile.raw_text else None,
            numeric_recall=0.0 if profile.raw_text else None,
            numeric_precision=0.0 if profile.raw_text else None,
            table_shape_pass=False,
            table_count=0,
            failures=["empty_parser_output"],
        )

    shape_result = _table_shape(value)
    shape_pass = shape_result.pass_
    table_count = shape_result.table_count
    source_is_reliable = (
        profile.page_class.value == "native_text"
        and bool(profile.raw_text and profile.raw_text.strip())
    )
    source_text_recall: Optional[float] = None
    source_text_precision: Optional[float] = None
    numeric_recall: Optional[float] = None
    numeric_precision: Optional[float] = None
    failures: List[str] = []
    warnings: List[str] = list(shape_result.soft_findings)
    source_numbers: Counter = Counter()  # computed inside source_is_reliable block
    missing_tokens: Counter = Counter()  # computed once, reused

    if source_is_reliable:
        source_tokens = token_multiset(profile.raw_text)
        if not source_tokens:
            warnings.append("source_text_empty_after_tokenization")
            source_is_reliable = False

    if source_is_reliable:
        source_numbers = number_multiset(profile.raw_text)
        predicted_tokens = token_multiset(value)
        missing_tokens = source_tokens - predicted_tokens  # compute once, reuse below
        source_text_recall = multiset_recall(source_tokens, predicted_tokens)
        source_text_precision = (
            multiset_precision(source_tokens, predicted_tokens)
            if predicted_tokens else None
        )
        predicted_numbers = number_multiset(value)
        numeric_recall = (
            multiset_recall(source_numbers, predicted_numbers)
            if source_numbers else None
        )
        numeric_precision = (
            multiset_precision(source_numbers, predicted_numbers)
            if predicted_numbers else None
        )

        source_num_contexts = extract_numbers_with_context(profile.raw_text)
        source_financial = {
            n_val for n_val, p_ctx, n_ctx, _ in source_num_contexts
            if classify_number(n_val, p_ctx, n_ctx) == "financial"
        }
        pred_num_contexts = extract_numbers_with_context(value)
        predicted_financial = {
            n_val for n_val, p_ctx, n_ctx, _ in pred_num_contexts
            if classify_number(n_val, p_ctx, n_ctx) == "financial"
        }

        # Cross-page boundary recall resolution (text and numbers):
        # Bug 3 fix: Only resolve genuine alphabetic tokens (not stopwords).
        # Reject-all: If candidate tokens exceed cap, reject resolution completely to prevent masking loss.
        # Cross-page recall resolution: reject-all semantics when candidate > cap.
        # Rationale: recall ưu tiên an toàn — nếu truncate, loss > cap → false PASS.
        if source_tokens and source_text_recall is not None and source_text_recall < config.min_docling_text_recall:
            resolvable_missing = Counter({
                tok: cnt for tok, cnt in missing_tokens.items()
                if is_resolvable_token(tok)
            })
            adjacent_tokens = Counter()
            if prev_page_markdown:
                adjacent_tokens |= token_multiset(prev_page_markdown)
            if next_page_markdown:
                adjacent_tokens |= token_multiset(next_page_markdown)

            candidate_resolved = resolvable_missing & adjacent_tokens
            total_candidate = sum(candidate_resolved.values())
            cap = max(int(config.max_cross_page_token_ratio * sum(source_tokens.values())), 3)

            # Reject-all when candidate exceeds cap
            if 0 < total_candidate <= cap:
                adj_tokens = predicted_tokens + candidate_resolved
                adj_text_rec = multiset_recall(source_tokens, adj_tokens)
                if adj_text_rec >= config.min_docling_text_recall:
                    warnings.append(f"cross_page_boundary_resolved_text_recall:{adj_text_rec:.3f}")
                    source_text_recall = adj_text_rec

        if source_numbers and numeric_recall is not None and numeric_recall < config.min_docling_numeric_recall:
            missing = source_numbers - predicted_numbers
            # Filter: Only allow resolving financial numbers (discard metadata/year/ordinals)
            resolvable_missing_nums = Counter({
                n_val: missing[n_val] for n_val in missing if n_val in source_financial
            })

            adjacent_numbers = Counter()
            if prev_page_markdown:
                adjacent_numbers |= number_multiset(prev_page_markdown)
            if next_page_markdown:
                adjacent_numbers |= number_multiset(next_page_markdown)

            candidate_num_resolved = resolvable_missing_nums & adjacent_numbers
            total_num_candidate = sum(candidate_num_resolved.values())
            num_cap = max(int(config.max_cross_page_token_ratio * sum(source_numbers.values())), 3)

            # Reject-all when candidate numbers exceed cap
            if 0 < total_num_candidate <= num_cap:
                adj_predicted_numbers = predicted_numbers + candidate_num_resolved
                adj_recall = multiset_recall(source_numbers, adj_predicted_numbers)
                if adj_recall >= config.min_docling_numeric_recall:
                    warnings.append(f"cross_page_boundary_resolved_numeric_recall:{adj_recall:.3f}")
                    numeric_recall = adj_recall

        # ---------- Text Precision Gate (Bug 6) ----------
        # Parser hallucinate text -> recall cao nhưng precision thấp.
        # Mirror logic của numeric precision: cho phép exemption cho token
        # xuất hiện ở trang kề (benefit of the doubt), nhưng cap 5%.
        if (
            predicted_tokens
            and source_text_precision is not None
            and source_text_precision < config.min_docling_text_precision
        ):
            extra_tokens = predicted_tokens - source_tokens
            resolvable_extra = Counter({
                tok: cnt for tok, cnt in extra_tokens.items()
                if is_resolvable_token(tok)
            })

            adjacent_tokens = Counter()
            if prev_page_markdown:
                adjacent_tokens |= token_multiset(prev_page_markdown)
            if next_page_markdown:
                adjacent_tokens |= token_multiset(next_page_markdown)

            overflow_candidates = sum((resolvable_extra & adjacent_tokens).values())
            cap = max(int(config.max_cross_page_token_ratio * sum(predicted_tokens.values())), 3)
            exempt = min(overflow_candidates, cap)

            common = source_tokens & predicted_tokens
            adj_predicted_total = max(1, sum(predicted_tokens.values()) - exempt)
            adj_precision = sum(common.values()) / adj_predicted_total

            if adj_precision >= config.min_docling_text_precision:
                warnings.append(f"text_precision_adjusted:{adj_precision:.3f}")
                source_text_precision = adj_precision
            elif sum(extra_tokens.values()) <= 3:
                # Fix: check if extra tokens contain actual numbers.
                # Original had type mismatch: extra_tokens.keys() are text tokens,
                # predicted_financial are normalized number strings — intersection always empty.
                extra_has_numbers = any(NUMBER_PATTERN.search(tok) for tok in extra_tokens)
                if not extra_has_numbers:
                    warnings.append(f"text_precision_minor_delta:{source_text_precision:.3f}")
                else:
                    failures.append(f"text_precision_below_gate:{source_text_precision:.3f}")
            else:
                failures.append(
                    f"text_precision_below_gate:{source_text_precision:.3f}"
                )

        # Precision adjustment for repeated table year headers & bounded cross-page overflow:
        # Repeating column fiscal year headers (e.g. 2025, 2024) across sub-tables
        # is structurally beneficial and must not fail the precision gate.
        if (
            predicted_numbers
            and numeric_precision is not None
            and numeric_precision < config.min_docling_numeric_precision
            and numeric_recall is not None
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
        # A page is narrative when text recall is high, numeric density is low,
        # and table count is small.
        num_density = sum(source_numbers.values()) / max(1, sum(source_tokens.values()))
        is_narrative_page = (
            source_text_recall is not None
            and source_text_recall >= config.min_narrative_text_recall
            and num_density < 0.20
            and (
                table_count == 0
                or (table_count <= 1 and sum(source_numbers.values()) <= config.max_narrative_numbers)
            )
        )

        if source_text_recall is not None and source_text_recall < config.min_docling_text_recall:
            if sum(missing_tokens.values()) <= 3 and not (source_financial & set(missing_tokens.keys())):
                warnings.append(f"source_text_recall_minor_delta:{source_text_recall:.3f}")
            else:
                failures.append(f"source_text_recall_below_gate:{source_text_recall:.3f}")

        # Bug 7: Differentiate financial numbers vs metadata numbers on narrative and non-narrative pages
        if source_numbers and numeric_recall is not None and numeric_recall < config.min_docling_numeric_recall:
            missing_critical = source_financial - set(predicted_numbers.keys())
            if is_narrative_page:
                if missing_critical:
                    failures.append(f"narrative_financial_missing:{numeric_recall:.3f}")
                else:
                    warnings.append(f"narrative_minor_numeric_delta:recall={numeric_recall:.3f}")
            else:  # non-narrative
                if missing_critical:
                    failures.append(f"financial_recall_below_gate:{numeric_recall:.3f}")
                else:
                    warnings.append(f"numeric_recall_metadata_only:{numeric_recall:.3f}")

        if predicted_numbers and numeric_precision is not None and numeric_precision < config.min_docling_numeric_precision:
            extra_critical = predicted_financial - set(source_numbers.keys())
            if is_narrative_page:
                if extra_critical:
                    failures.append(f"narrative_financial_hallucinated:{numeric_precision:.3f}")
                else:
                    warnings.append(f"narrative_minor_numeric_delta:precision={numeric_precision:.3f}")
            else:  # non-narrative
                if extra_critical:
                    failures.append(f"financial_precision_below_gate:{numeric_precision:.3f}")
                else:
                    warnings.append(f"numeric_precision_metadata_only:{numeric_precision:.3f}")
    else:
        warnings.append("no_reliable_native_text_oracle_for_fidelity_check")

    if not shape_pass:
        # Only hard structure findings cause FAIL — soft findings are already in warnings.
        failures.extend(item for item in shape_result.hard_findings if item not in failures)
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
        source_text_precision=round(source_text_precision, 4) if source_text_precision is not None else None,
        numeric_recall=round(numeric_recall, 4) if numeric_recall is not None else None,
        numeric_precision=round(numeric_precision, 4) if numeric_precision is not None else None,
        table_shape_pass=shape_pass,
        table_count=table_count,
        warnings=warnings,
        failures=failures,
    )
