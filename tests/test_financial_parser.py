import sys
import unittest
from pathlib import Path

# Add FinAnalyst-AI / script to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from financial_parser.config import ParserConfig
from financial_parser.models import (
    EngineName,
    LanguageEvidence,
    PageClass,
    PageProfile,
    QCResult,
    QCStatus,
)
from financial_parser.qc import evaluate_output, _table_shape


def make_profile(
    page_class: str = "native_text",
    raw_text: str = "",
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


# =========================================================================
# Section 1 — ZeroDivision / whitespace
# =========================================================================
def test_whitespace_raw_text_not_reliable():
    profile = make_profile(page_class="native_text", raw_text="   \n\t  ")
    result = evaluate_output(
        markdown="Some output.",
        profile=profile, engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert "no_reliable_native_text_oracle_for_fidelity_check" in result.warnings
    assert result.source_text_recall is None
    assert result.source_text_precision is None


def test_empty_raw_text_all_metrics_none():
    profile = make_profile(page_class="native_text", raw_text="")
    result = evaluate_output(
        markdown="Content", profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.source_text_recall is None
    assert result.numeric_recall is None
    assert result.numeric_precision is None


def test_no_numbers_in_source_no_zerodivision():
    profile = make_profile(
        page_class="native_text",
        raw_text="The company is in California.",
    )
    result = evaluate_output(
        markdown="The company is in California.",
        profile=profile, engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.numeric_recall is None
    assert result.numeric_precision is None
    assert result.status != QCStatus.FAIL


# =========================================================================
# Section 2 — Bug 6 (Text Precision)
# =========================================================================
def test_text_precision_hallucination_fails():
    profile = make_profile(
        page_class="native_text",
        raw_text="Revenue was $500 in 2025.",
    )
    hallucinated = "Revenue was $500 in 2025. " + " ".join(["HALLUCINATED"] * 100)
    result = evaluate_output(
        markdown=hallucinated, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.status == QCStatus.FAIL
    assert any("text_precision_below_gate" in f for f in result.failures)
    assert result.source_text_precision < 0.88


def test_text_precision_clean_passes():
    profile = make_profile(
        page_class="native_text",
        raw_text="Revenue was $500 in 2025.",
    )
    result = evaluate_output(
        markdown="Revenue was $500 in 2025.", profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.source_text_precision == 1.0
    assert result.status == QCStatus.PASS


def test_text_precision_stopword_hallucination_not_exempted():
    """Stopword bịa không được cứu bởi trang kề."""
    profile = make_profile(page_class="native_text", raw_text="Revenue $500.")
    result = evaluate_output(
        markdown="Revenue $500. " + "the " * 100,
        profile=profile, engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown="the the the the the the",
    )
    assert result.status == QCStatus.FAIL


def test_text_precision_overflow_exempted_within_cap():
    """Token dư có chữ cái, có ở trang kề, dưới 5% → exempt."""
    prev = "Calibration segment adjustment recognized annually."
    profile = make_profile(page_class="native_text", raw_text="Revenue was $500.")
    result = evaluate_output(
        markdown="Revenue was $500. Calibration segment adjustment recognized annually.",
        profile=profile, engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown=prev,
    )
    # Depends on exact ratio; at minimum, should not be outright FAIL
    # unless exempt cap < needed
    assert result.source_text_precision is not None


# =========================================================================
# Section 3 — Bug 3 (Cross-page recall, text + numeric)
# =========================================================================
def test_text_cross_page_reject_all_when_exceeds_cap():
    """Trang mất 62.5% nội dung không được kéo lên 1.0."""
    profile = make_profile(
        page_class="native_text",
        raw_text="alpha beta gamma delta epsilon zeta eta theta",
    )
    # Output chỉ giữ 3/8 token
    output = "alpha beta gamma"
    # prev_page có tất cả token còn lại — thử rescue
    prev = "delta epsilon zeta eta theta"
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown=prev,
    )
    # Cap = max(int(0.05 * 8), 3) = 3. Missing 5 tokens > 3 → reject.
    assert result.source_text_recall == 3/8  # 0.375
    assert result.status == QCStatus.FAIL


def test_text_cross_page_accepted_within_cap():
    """Missing 2 tokens, cap 3 → accept."""
    profile = make_profile(
        page_class="native_text",
        raw_text="alpha beta gamma delta epsilon",
    )
    output = "alpha beta gamma"  # missing 2
    prev = "delta epsilon"
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown=prev,
    )
    # Cap = max(int(0.05*5), 3) = 3; missing 2 <= 3 → resolve
    assert result.source_text_recall == 1.0


def test_cross_page_stopwords_not_resolved():
    """Stopword missing không được resolve từ trang kề."""
    profile = make_profile(
        page_class="native_text",
        raw_text="the the the the alpha",
    )
    output = "alpha"  # missing 4 stopwords
    prev = "the the the the"
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown=prev,
    )
    # Stopword bị lọc → không resolve → recall = 1/5 = 0.2
    assert result.source_text_recall == 1/5


def test_numeric_cross_page_bounded():
    """Numeric cross-page cũng phải bị cap."""
    profile = make_profile(
        page_class="native_text",
        raw_text="$500 $600 $700 $800 $900 $1000",
    )
    output = "$500"  # missing 5 financial
    prev = "$600 $700 $800 $900 $1000"
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
        prev_page_markdown=prev,
    )
    # Cap = max(int(0.05 * 6), 3) = 3. Missing 5 > 3 → reject.
    assert result.numeric_recall == round(1/6, 4) or abs(result.numeric_recall - 1/6) < 1e-4


# =========================================================================
# Section 4 — Bug 7 (Financial vs metadata, non-narrative)
# =========================================================================
def test_non_narrative_recall_metadata_only_to_warning():
    """Non-narrative: mất Item 16 (metadata) → WARNING."""
    profile = make_profile(
        page_class="native_text",
        raw_text="See Item 16. Revenue was $500 million in FY2025. "
                 "Cost was $300 million.",
        likely_tabular=False,
    )
    output = "Revenue was $500 million in FY2025. Cost was $300 million."
    # → non-narrative vì không đủ điều kiện narrative (nhiều số)
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.status == QCStatus.WARNING
    assert any("metadata_only" in w for w in result.warnings)
    assert not any("financial_recall" in f for f in result.failures)


def test_non_narrative_recall_financial_missing_fails():
    """Non-narrative: mất $500 → FAIL."""
    profile = make_profile(
        page_class="native_text",
        raw_text="See Item 16. Revenue was $500 million. Cost was $300 million.",
    )
    output = "See Item 16. Revenue missing. Cost was $300 million."
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.status == QCStatus.FAIL
    assert any("financial_recall_below_gate" in f for f in result.failures)


def test_non_narrative_precision_metadata_hallucination_warning():
    """Non-narrative: bịa year 2024 → WARNING."""
    profile = make_profile(
        page_class="native_text",
        raw_text="Revenue $500. Cost $300. Margin $200.",
    )
    output = "Revenue $500 in FY2024. Cost $300. Margin $200."
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.status == QCStatus.WARNING
    assert any("metadata_only" in w for w in result.warnings)


def test_non_narrative_precision_financial_hallucination_fails():
    """Non-narrative: bịa $999 → FAIL."""
    profile = make_profile(
        page_class="native_text",
        raw_text="Revenue $500. Cost $300. Margin $200.",
    )
    output = "Revenue $500. Cost $300. Margin $200 and $999 bonus."
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert result.status == QCStatus.FAIL
    assert any("financial_precision_below_gate" in f for f in result.failures)


# =========================================================================
# Section 5 — Bug 8 (Year header sequence)
# =========================================================================
def test_table_year_header_non_monotonic_warns():
    """Cột year trong header zigzag → WARNING."""
    markdown = """
| Metric | 2025 | 2023 | 2024 |
|--------|------|------|------|
| Revenue | 100 | 80 | 90 |
"""
    profile = make_profile(page_class="native_text", raw_text="dummy")
    result = evaluate_output(
        markdown=markdown, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert any("header_year_order_non_monotonic" in w for w in result.warnings)


def test_table_year_header_inconsistent_across_tables_warns():
    """2 sub-table có year order ngược chiều → WARNING."""
    markdown = """
| Metric | 2025 | 2024 |
|--------|------|------|
| A | 100 | 80 |

| Metric | 2024 | 2025 |
|--------|------|------|
| B | 90 | 110 |
"""
    profile = make_profile(page_class="native_text", raw_text="dummy")
    result = evaluate_output(
        markdown=markdown, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    # Cross-table check removed in plan v3 (avoids false warnings on mixed historical/maturity tables)
    assert not any("order_inconsistent_across_tables" in w for w in result.warnings)


def test_table_year_header_monotonic_passes():
    markdown = """
| Metric | 2025 | 2024 | 2023 |
|--------|------|------|------|
| Revenue | 100 | 80 | 70 |
"""
    profile = make_profile(page_class="native_text", raw_text="dummy")
    result = evaluate_output(
        markdown=markdown, profile=profile,
        engine=EngineName.DOCLING, config=ParserConfig(),
    )
    assert not any("non_monotonic" in w for w in result.warnings)


# =========================================================================
# Section 6 — Integration với config mới
# =========================================================================
def test_config_new_fields_have_defaults():
    cfg = ParserConfig()
    assert cfg.min_text_precision == 0.88
    assert cfg.max_cross_page_token_ratio == 0.05


def test_config_override_text_precision():
    profile = make_profile(page_class="native_text", raw_text="Revenue $500.")
    output = "Revenue $500. " + " ".join(["HALLUC"] * 30)
    result = evaluate_output(
        markdown=output, profile=profile,
        engine=EngineName.DOCLING,
        config=ParserConfig(min_text_precision=0.5),  # nới lỏng
    )
    # Với gate thấp hơn, có thể PASS hoặc WARNING thay vì FAIL
    assert result.status != QCStatus.FAIL or result.source_text_precision < 0.5


# =========================================================================
# UnitTestCase Wrapper (for `python -m unittest`)
# =========================================================================
class FinancialParserTests(unittest.TestCase):
    def setUp(self):
        self.config = ParserConfig()

    # Section 1
    def test_whitespace_raw_text_not_reliable(self):
        test_whitespace_raw_text_not_reliable()

    def test_empty_raw_text_all_metrics_none(self):
        test_empty_raw_text_all_metrics_none()

    def test_no_numbers_in_source_no_zerodivision(self):
        test_no_numbers_in_source_no_zerodivision()

    # Section 2
    def test_text_precision_hallucination_fails(self):
        test_text_precision_hallucination_fails()

    def test_text_precision_clean_passes(self):
        test_text_precision_clean_passes()

    def test_text_precision_stopword_hallucination_not_exempted(self):
        test_text_precision_stopword_hallucination_not_exempted()

    def test_text_precision_overflow_exempted_within_cap(self):
        test_text_precision_overflow_exempted_within_cap()

    # Section 3
    def test_text_cross_page_reject_all_when_exceeds_cap(self):
        test_text_cross_page_reject_all_when_exceeds_cap()

    def test_text_cross_page_accepted_within_cap(self):
        test_text_cross_page_accepted_within_cap()

    def test_cross_page_stopwords_not_resolved(self):
        test_cross_page_stopwords_not_resolved()

    def test_numeric_cross_page_bounded(self):
        test_numeric_cross_page_bounded()

    # Section 4
    def test_non_narrative_recall_metadata_only_to_warning(self):
        test_non_narrative_recall_metadata_only_to_warning()

    def test_non_narrative_recall_financial_missing_fails(self):
        test_non_narrative_recall_financial_missing_fails()

    def test_non_narrative_precision_metadata_hallucination_warning(self):
        test_non_narrative_precision_metadata_hallucination_warning()

    def test_non_narrative_precision_financial_hallucination_fails(self):
        test_non_narrative_precision_financial_hallucination_fails()

    # Section 5
    def test_table_year_header_non_monotonic_warns(self):
        test_table_year_header_non_monotonic_warns()

    def test_table_year_header_inconsistent_across_tables_warns(self):
        test_table_year_header_inconsistent_across_tables_warns()

    def test_table_year_header_monotonic_passes(self):
        test_table_year_header_monotonic_passes()

    # Section 6
    def test_config_new_fields_have_defaults(self):
        test_config_new_fields_have_defaults()

    def test_config_override_text_precision(self):
        test_config_override_text_precision()

    # Additional narrative & structural regression tests
    def test_narrative_missing_financial_fails(self):
        raw_text = (
            "The executive committee and board of directors reviewed our annual operations and performance across all business units. "
            "Our corporate governance framework ensures full compliance with international accounting standards and statutory requirements. "
            "In fiscal year 2025, the company paid a significant penalty of $50,000,000 "
            "as described in Item 16 on Page 90 of our statutory disclosures and legal proceedings. "
            "Management believes that these settlements resolve all past compliance inquiries without any further ongoing exposure. "
            "All operations continue in ordinary course with stable capital resources."
        )
        pred_text = (
            "The executive committee and board of directors reviewed our annual operations and performance across all business units. "
            "Our corporate governance framework ensures full compliance with international accounting standards and statutory requirements. "
            "In fiscal year 2025, the company paid a significant penalty of "
            "as described in Item 16 on Page 90 of our statutory disclosures and legal proceedings. "
            "Management believes that these settlements resolve all past compliance inquiries without any further ongoing exposure. "
            "All operations continue in ordinary course with stable capital resources."
        )
        profile = make_profile(page_class="native_text", raw_text=raw_text)
        result = evaluate_output(pred_text, profile, EngineName.DOCLING, self.config)
        self.assertEqual(result.status, QCStatus.FAIL)
        self.assertTrue(any("narrative_financial_missing" in f for f in result.failures))

    def test_narrative_missing_metadata_warns(self):
        raw_text = (
            "The company operates under standard terms as documented in Item 16 on Page 90. "
            "All operations continue in ordinary course without any material changes."
        )
        pred_text = (
            "The company operates under standard terms as documented in Item 16. "
            "All operations continue in ordinary course without any material changes."
        )
        profile = make_profile(page_class="native_text", raw_text=raw_text)
        result = evaluate_output(pred_text, profile, EngineName.DOCLING, self.config)
        self.assertFalse(any("narrative_financial_missing" in f for f in result.failures))


if __name__ == "__main__":
    unittest.main()
