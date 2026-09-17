from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from financial_parser.canonical import markdown_to_canonical_page
from financial_parser.config import ParserConfig
from financial_parser.markdown_utils import extract_markdown_tables, sanitize_markdown
from financial_parser.models import EngineName, LanguageEvidence, PageClass, PageProfile, QCStatus, RouteDecision
from financial_parser.qc import evaluate_output


def native_profile(raw_text: str) -> PageProfile:
    return PageProfile(
        pdf_page=1,
        page_class=PageClass.NATIVE_TEXT,
        language=LanguageEvidence(primary="vi", confidence=0.9),
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
    )


class FinancialParserUnitTests(unittest.TestCase):
    def test_sanitizer_preserves_financial_states(self) -> None:
        result = sanitize_markdown("```markdown\n| A | B |\n|---|---|\n| Lỗ | (1.234) |\n| Nil | - |\n```")
        self.assertIn("(1.234)", result)
        self.assertIn("| Nil | - |", result)

    def test_table_parser_handles_escaped_pipe(self) -> None:
        tables = extract_markdown_tables("| Label | Value |\n|---|---|\n| A \\| B | 100 |")
        self.assertEqual(tables[0]["rows"][0], ["A | B", "100"])

    def test_docling_qc_accepts_matching_native_text(self) -> None:
        raw = "Tiền mặt 1.200 300"
        markdown = "| Khoản mục | Cuối năm | Đầu năm |\n|---|---|---|\n| Tiền mặt | 1.200 | 300 |"
        qc = evaluate_output(markdown, native_profile(raw), EngineName.DOCLING, ParserConfig())
        self.assertEqual(qc.status, QCStatus.PASS)

    def test_docling_qc_rejects_broken_table_shape(self) -> None:
        raw = "Tiền mặt 1.200 300"
        markdown = "| Khoản mục | Cuối năm | Đầu năm |\n|---|---|---|\n| Tiền mặt | 1.200 300 |"
        qc = evaluate_output(markdown, native_profile(raw), EngineName.DOCLING, ParserConfig())
        self.assertEqual(qc.status, QCStatus.FAIL)

    def test_docling_qc_rejects_two_numbers_in_one_numeric_cell(self) -> None:
        raw = "Vốn 2024 24 25"
        markdown = "| Khoản mục | Năm nay | Năm trước |\n|---|---|---|\n| Vốn | 2024 24 | 25 |"
        qc = evaluate_output(markdown, native_profile(raw), EngineName.DOCLING, ParserConfig())
        self.assertEqual(qc.status, QCStatus.FAIL)

    def test_canonical_table_has_provenance(self) -> None:
        profile = native_profile("Tiền mặt 1.200 300")
        qc = evaluate_output(
            "| Khoản mục | Cuối năm | Đầu năm |\n|---|---|---|\n| Tiền mặt | 1.200 | 300 |",
            profile,
            EngineName.DOCLING,
            ParserConfig(),
        )
        canonical = markdown_to_canonical_page(
            markdown="## 1. TIỀN\n\n| Khoản mục | Cuối năm | Đầu năm |\n|---|---|---|\n| Tiền mặt | 1.200 | 300 |",
            document_id="test",
            pdf_page=1,
            printed_page=1,
            profile=profile,
            route=RouteDecision(1, EngineName.DOCLING, ["unit_test"]),
            qc=qc,
        )
        table = next(block for block in canonical["page"]["blocks"] if block["block_type"] == "table")
        self.assertEqual(table["table_id"], "p1_t01")
        self.assertEqual(table["source_refs"], [{"pdf_page": 1}])

    def test_deepseek_config_and_factory(self) -> None:
        from financial_parser.engines import DeepSeekVisionEngine, create_vlm_engine

        config = ParserConfig(
            vlm_provider="deepseek",
            deepseek_api_key="sk-test-key",
            deepseek_model="deepseek-flash",
        )
        self.assertEqual(config.vlm_provider, "deepseek")
        self.assertEqual(config.deepseek_model, "deepseek-flash")
        engine = create_vlm_engine(config)
        self.assertIsInstance(engine, DeepSeekVisionEngine)
        self.assertTrue(engine.available(config))

    def test_sec_10k_section_and_note_headers(self) -> None:
        from financial_parser.canonical import _heading_metadata

        code, title = _heading_metadata("Item 8. Financial Statements and Supplementary Data")
        self.assertEqual(code, "8")
        self.assertEqual(title, "Item 8 Financial Statements and Supplementary Data")

        code, title = _heading_metadata("Note 1 – Summary of Significant Accounting Policies")
        self.assertEqual(code, "1")
        self.assertEqual(title, "Note 1 Summary of Significant Accounting Policies")

        code, title = _heading_metadata("Item 1A. Risk Factors")
        self.assertEqual(code, "1A")
        self.assertEqual(title, "Item 1A Risk Factors")

    def test_sec_10k_unit_and_currency_detection(self) -> None:
        from financial_parser.canonical import _unit_and_period

        unit, _ = _unit_and_period("# Statements of Operations\n(In millions)\nForm 10-K")
        self.assertIsNotNone(unit)
        self.assertEqual(unit["scale"], "million")
        self.assertEqual(unit["currency"], "USD")
        self.assertEqual(unit["raw"], "(In millions)")


if __name__ == "__main__":
    unittest.main()

