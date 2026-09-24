"""Canonical filesystem locations. Import this instead of hard-coding CWD-relative paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
APPS_DIR = PROJECT_ROOT / "apps"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUT_PARSING = OUTPUTS_DIR / "parsing"
OUTPUT_CHUNKING = OUTPUTS_DIR / "chunking"
OUTPUT_RETRIEVAL = OUTPUTS_DIR / "retrieval"
OUTPUT_DEMO = OUTPUTS_DIR / "demo"
OUTPUT_REVIEW = OUTPUTS_DIR / "review"
OUTPUT_PARSER_DEFAULT = OUTPUTS_DIR / "financial_parser"
OUTPUT_TEST = OUTPUTS_DIR / "test"

TEMP_DEMO_DIR = DATA_DIR / "temp_demo"
GOLD_TEST_SET_DIR = DATA_DIR / "gold_test_set"
GOLD_TEST_SET_FILE = GOLD_TEST_SET_DIR / "v0_gold_questions.jsonl"
SEC_FILINGS_DIR = DATA_DIR / "sec_filings"
VIETNAMESE_BCTC_DIR = DATA_DIR / "vietnamese_bctc"
