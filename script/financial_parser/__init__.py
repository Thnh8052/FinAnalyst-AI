"""Financial-report parsing package for VAS, IFRS and SEC filings."""

from .config import ParserConfig
from .router import FinancialReportRouter

__all__ = ["FinancialReportRouter", "ParserConfig"]
