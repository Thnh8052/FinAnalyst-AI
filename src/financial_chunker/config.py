"""
Configuration for Financial Chunker.
Auto-detects parsed company directories to construct COMPANY_SPECS.
"""

from pathlib import Path
from typing import Dict, List, Any

try:
    from finanalyst.paths import OUTPUT_CHUNKING, OUTPUT_PARSING
except ImportError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_PARSING = _PROJECT_ROOT / "outputs" / "parsing"
    OUTPUT_CHUNKING = _PROJECT_ROOT / "outputs" / "chunking"

OUTPUT_PARSING_ROOT = OUTPUT_PARSING
OUTPUT_CHUNKING_ROOT = OUTPUT_CHUNKING

TICKER_MAP = {
    "3M": "MMM",
    "ACTIVISIONBLIZZARD": "ATVI",
    "ADOBE": "ADBE",
    "AES": "AES",
    "AMAZON": "AMZN",
    "AMCOR": "AMCR",
    "AMD": "AMD",
    "AMERICANWATERWORKS": "AWK",
    "APPLE": "AAPL",
    "BESTBUY": "BBY",
    "BOEING": "BA",
    "BOSTONPROPERTIES": "BXP",
    "COCACOLA": "KO",
    "CORNING": "GLW",
    "COSTCO": "COST",
    "CVSHEALTH": "CVS",
    "EBAY": "EBAY",
    "GENERALMILLS": "GIS",
    "INTEL": "INTC",
    "JOHNSON_JOHNSON": "JNJ",
    "JPMORGAN": "JPM",
    "KRAFTHEINZ": "KHC",
    "LOCKHEEDMARTIN": "LMT",
    "MCDONALDS": "MCD",
    "MGMRESORTS": "MGM",
    "MICROSOFT": "MSFT",
    "NETFLIX": "NFLX",
    "NIKE": "NKE",
    "NVIDIA": "NVDA",
    "ORACLE": "ORCL",
    "PEPSICO": "PEP",
    "PFIZER": "PFE",
    "PG_E": "PCG",
    "SALESFORCE": "CRM",
    "ULTABEAUTY": "ULTA",
    "VERIZON": "VZ",
    "WALMART": "WMT",
}

BENCHMARK_DIRS = {
    "amazon_2024_10k",
    "amd_2025_10k",
    "apple_2024_10k",
    "intel_2024_10k",
    "nike_2025_10k",
    "nvidia_2025_10k",
    "walmart_2024_10k",
}

def get_company_specs(language: str = "en", benchmark_only: bool = False) -> List[Dict[str, Any]]:
    """
    Auto-detect parsed companies in output_parsing.
    language: 'en' for US dataset (10k), 'vi' for Vietnamese dataset (bctc).
    benchmark_only: If True, returns the 7 representative 10-K filings (most recent year per company).
    """
    specs = []
    if not OUTPUT_PARSING_ROOT.exists():
        return specs
        
    for d in OUTPUT_PARSING_ROOT.iterdir():
        if d.is_dir():
            dir_name = d.name
            pages_dir = d / "pages"
            page_count = len(list(pages_dir.glob("page_*.json"))) if pages_dir.exists() else 0
            if page_count == 0:
                continue

            if benchmark_only and dir_name not in BENCHMARK_DIRS:
                continue

            # Extrapolate Ticker & Fiscal Year (robust token scanning)
            parts = dir_name.lower().split('_')
            year_idx = None
            for i in range(len(parts) - 1, -1, -1):
                p = parts[i]
                if p.isdigit() and len(p) == 4 and p.startswith(('19', '20')):
                    year_idx = i
                    break

            if year_idx is not None:
                company_key = "_".join(parts[:year_idx]).upper()
                fiscal_year = int(parts[year_idx])
            else:
                company_key = parts[0].upper()
                fiscal_year = 2024

            ticker = (
                TICKER_MAP.get(company_key)
                or TICKER_MAP.get(company_key.replace('_', ''))
                or company_key
            )

            specs.append({
                "dir_name": dir_name,
                "ticker": ticker,
                "name": dir_name,
                "pages": page_count,
                "fiscal_year": fiscal_year,
            })
                
    # Sort for deterministic ordering
    specs.sort(key=lambda x: x["dir_name"])
    return specs

# Default export for scripts
COMPANY_SPECS = get_company_specs(language="en", benchmark_only=False)
BENCHMARK_COMPANY_SPECS = get_company_specs(language="en", benchmark_only=True)
