"""Tests for _table_shape refactor: hard/soft separation, adjacent dedup, year extraction."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from financial_parser.config import ParserConfig
from financial_parser.models import (
    EngineName,
    LanguageEvidence,
    PageClass,
    PageProfile,
    QCStatus,
)
from financial_parser.qc import (
    TableShapeResult,
    _adjacent_dedup,
    _extract_header_years,
    _table_shape,
    evaluate_output,
)


def make_profile(
    raw_text: str = "",
    page_class: str = "native_text",
    likely_tabular: bool = False,
) -> PageProfile:
    return PageProfile(
        pdf_page=1,
        page_class=PageClass(page_class),
        language=LanguageEvidence(primary="en", confidence=0.95),
        raw_text=raw_text,
        char_count=len(raw_text),
        word_count=len(raw_text.split()),
        text_density=0.1,
        image_count=0,
        max_image_coverage=0.0,
        total_image_coverage=0.0,
        replacement_char_ratio=0.0,
        private_use_ratio=0.0,
        control_char_ratio=0.0,
        type3_font_count=0,
        has_suspicious_font=False,
        likely_tabular=likely_tabular,
    )


# ===========================================================================
# _adjacent_dedup
# ===========================================================================
class TestAdjacentDedup(unittest.TestCase):

    def test_empty_guard(self):
        """Empty input must not raise IndexError."""
        assert _adjacent_dedup([]) == []

    def test_single_element(self):
        assert _adjacent_dedup([2025]) == [2025]

    def test_all_same(self):
        assert _adjacent_dedup([2025, 2025, 2025]) == [2025]

    def test_adjacent_pairs(self):
        assert _adjacent_dedup([2025, 2025, 2024, 2024]) == [2025, 2024]

    def test_triple_adjacent(self):
        assert _adjacent_dedup([2025, 2025, 2025, 2024]) == [2025, 2024]

    def test_non_adjacent_preserved(self):
        """Non-adjacent duplicates are NOT deduped - they may be valid sub-metrics."""
        assert _adjacent_dedup([2025, 2024, 2025]) == [2025, 2024, 2025]

    def test_no_duplicates(self):
        assert _adjacent_dedup([2025, 2024, 2023]) == [2025, 2024, 2023]

    def test_ascending(self):
        assert _adjacent_dedup([2023, 2024, 2025]) == [2023, 2024, 2025]

    def test_range_cell_fallback(self):
        assert _adjacent_dedup([2024, 2025, 2023, 2024]) == [2024, 2025, 2023, 2024]


# ===========================================================================
# _extract_header_years
# ===========================================================================
class TestExtractHeaderYears(unittest.TestCase):

    def test_normal_years(self):
        assert _extract_header_years(["Item", "2025", "2024"]) == [2025, 2024]

    def test_fy_prefix(self):
        """FY2025 must match - digit boundary, not word boundary."""
        assert _extract_header_years(["Metric", "FY2025", "FY2024"]) == [2025, 2024]

    def test_quarter_prefix(self):
        assert _extract_header_years(["Q1 2025", "Q1 2024"]) == [2025, 2024]

    def test_range_cell_skipped(self):
        """Range cell '2024-2025' must NOT produce two years."""
        assert _extract_header_years(["Metric", "2024-2025", "2023-2024"]) == []

    def test_range_cell_with_fy(self):
        """FY2024-FY2025 range also skipped."""
        assert _extract_header_years(["FY2024\u2013FY2025"]) == []

    def test_five_digit_rejected(self):
        """12025 should not extract 2025 due to digit boundary."""
        assert _extract_header_years(["Item", "12025"]) == []

    def test_no_years(self):
        assert _extract_header_years(["Item", "Value", "Notes"]) == []

    def test_mixed_year_and_text(self):
        assert _extract_header_years(["Revenue", "Year ended 2025", "Year ended 2024"]) == [2025, 2024]

    def test_year_in_parentheses(self):
        assert _extract_header_years(["Metric", "(2025)", "(2024)"]) == [2025, 2024]


# ===========================================================================
# Year monotonic check (via _table_shape soft findings)
# ===========================================================================
class TestYearMonotonic(unittest.TestCase):

    def _has_monotonic_warning(self, headers):
        """Build a minimal table and check for year order warning."""
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_line = "| " + " | ".join(["x"] * len(headers)) + " |"
        md = f"{header_line}\n{sep_line}\n{row_line}\n"
        result = _table_shape(md)
        return any("non_monotonic" in f for f in result.soft_findings)

    def test_zigzag_flagged(self):
        assert self._has_monotonic_warning(["M", "2024", "2025", "2023"])

    def test_zigzag_flagged_2(self):
        assert self._has_monotonic_warning(["M", "2025", "2023", "2024"])

    def test_monotonic_desc_pass(self):
        assert not self._has_monotonic_warning(["M", "2025", "2024", "2023"])

    def test_monotonic_inc_pass(self):
        assert not self._has_monotonic_warning(["M", "2023", "2024", "2025"])

    def test_simple_desc_pass(self):
        assert not self._has_monotonic_warning(["M", "2025", "2024"])

    def test_simple_inc_pass_limitation(self):
        """Documented limitation: simple reversal undetected."""
        assert not self._has_monotonic_warning(["M", "2024", "2025"])

    def test_adjacent_dup_deduped_then_pass(self):
        """[2025,2025,2024,2024] dedup -> [2025,2024] - monotonic desc, no flag."""
        assert not self._has_monotonic_warning(["M", "2025", "2025", "2024", "2024"])

    def test_single_year_no_check(self):
        assert not self._has_monotonic_warning(["M", "2025"])

    def test_no_years_no_check(self):
        assert not self._has_monotonic_warning(["Item", "Value", "Notes"])


# ===========================================================================
# Hard vs Soft separation
# ===========================================================================
class TestHardSoftSeparation(unittest.TestCase):

    def test_cell_count_mismatch_is_hard(self):
        md = "| A | B | C |\n|---|---|---|\n| 1 | 2 |\n"
        result = _table_shape(md)
        assert not result.pass_
        assert any("expected" in f for f in result.hard_findings)
        assert len(result.soft_findings) == 0

    def test_merged_numeric_cell_is_hard(self):
        md = "| Item | 2025 | 2024 |\n|---|---|---|\n| Shares | 11,237 28 | |\n"
        result = _table_shape(md)
        assert not result.pass_
        assert any("merged" in f for f in result.hard_findings)

    def test_year_non_monotonic_is_soft(self):
        md = "| Metric | 2024 | 2025 | 2023 |\n|---|---|---|---|\n| A | 1 | 2 | 3 |\n"
        result = _table_shape(md)
        assert result.pass_  # soft only -> pass_ = True
        assert any("non_monotonic" in f for f in result.soft_findings)
        assert len(result.hard_findings) == 0

    def test_less_than_two_columns_is_hard(self):
        md = "| A |\n|---|\n| 1 |\n"
        result = _table_shape(md)
        assert not result.pass_
        assert any("less_than_two" in f for f in result.hard_findings)

    def test_clean_table_all_pass(self):
        md = "| Metric | 2025 | 2024 |\n|---|---|---|\n| Revenue | 100 | 80 |\n| Profit | 50 | 40 |\n"
        result = _table_shape(md)
        assert result.pass_
        assert len(result.hard_findings) == 0
        assert len(result.soft_findings) == 0


# ===========================================================================
# NamedTuple backward compatibility
# ===========================================================================
class TestNamedTupleCompat(unittest.TestCase):

    def test_field_access(self):
        result = _table_shape("| A | B |\n|---|---|\n| 1 | 2 |")
        assert isinstance(result, TableShapeResult)
        assert result.pass_ is True
        assert result.table_count == 1
        assert isinstance(result.hard_findings, list)
        assert isinstance(result.soft_findings, list)

    def test_tuple_unpack(self):
        pass_, count, hard, soft = _table_shape("| A | B |\n|---|---|\n| 1 | 2 |")
        assert pass_ is True
        assert count == 1
        assert hard == []
        assert soft == []


# ===========================================================================
# evaluate_output integration
# ===========================================================================
class TestEvaluateOutputIntegration(unittest.TestCase):

    def setUp(self):
        self.config = ParserConfig()

    def test_hard_finding_produces_fail(self):
        md = "| A | B | C |\n|---|---|---|\n| 1 | 2 |\n"
        profile = make_profile(raw_text="A B C 1 2", page_class="native_text")
        result = evaluate_output(md, profile, EngineName.DOCLING, self.config)
        assert result.status == QCStatus.FAIL
        assert any("expected" in f for f in result.failures)

    def test_soft_finding_does_not_fail(self):
        """Year non-monotonic alone -> WARNING, not FAIL."""
        md = "| Metric | 2024 | 2025 | 2023 |\n|---|---|---|---|\n| Revenue | 100 | 200 | 50 |\n"
        profile = make_profile(
            raw_text="Metric 2024 2025 2023 Revenue 100 200 50",
            page_class="native_text",
        )
        result = evaluate_output(md, profile, EngineName.DOCLING, self.config)
        assert any("non_monotonic" in w for w in result.warnings)
        assert not any("non_monotonic" in f for f in result.failures)

    def test_no_double_count(self):
        """Structure findings must NOT appear in both warnings AND failures."""
        md = "| Item | 2025 | 2024 |\n|---|---|---|\n| Shares | 11,237 28 | |\n"
        profile = make_profile(
            raw_text="Item 2025 2024 Shares 11,237 28",
            page_class="native_text",
        )
        result = evaluate_output(md, profile, EngineName.DOCLING, self.config)
        merged_in_failures = [f for f in result.failures if "merged" in f]
        merged_in_warnings = [w for w in result.warnings if "merged" in w]
        assert len(merged_in_failures) >= 1
        assert len(merged_in_warnings) == 0


# ===========================================================================
# Cross-table check removed
# ===========================================================================
class TestCrossTableRemoved(unittest.TestCase):

    def test_mixed_direction_tables_no_flag(self):
        """Two tables with different year order on same page -> no warning."""
        md = (
            "| Metric | 2025 | 2024 |\n"
            "|---|---|---|\n"
            "| Revenue | 100 | 80 |\n"
            "\n"
            "| Maturity | 2026 | 2027 |\n"
            "|---|---|---|\n"
            "| Debt | 500 | 300 |\n"
        )
        result = _table_shape(md)
        assert "table_header_year_order_inconsistent_across_tables" not in result.soft_findings
        assert "table_header_year_order_inconsistent_across_tables" not in result.hard_findings


if __name__ == "__main__":
    unittest.main()
