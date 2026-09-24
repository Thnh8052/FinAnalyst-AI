"""
Configuration for Financial Chunker.
Auto-detects parsed company directories to construct COMPANY_SPECS.
"""

from pathlib import Path
from typing import Dict, List, Any
import re

from finanalyst.paths import OUTPUT_CHUNKING, OUTPUT_PARSING

OUTPUT_PARSING_ROOT = OUTPUT_PARSING
OUTPUT_CHUNKING_ROOT = OUTPUT_CHUNKING

def get_company_specs(language: str = "en", benchmark_only: bool = False) -> List[Dict[str, Any]]:
    """
    Auto-detect parsed companies in output_parsing.
    language: 'en' for US dataset (10k), 'vi' for Vietnamese dataset (bctc).
    benchmark_only: If True, returns the 7 representative 10-K filings (most recent year per company).
    """
    specs = []
    if not OUTPUT_PARSING_ROOT.exists():
        return specs
        
    TICKER_MAP = {
        "AMAZON": "AMZN",
        "AMD": "AMD",
        "APPLE": "AAPL",
        "INTEL": "INTC",
        "NIKE": "NKE",
        "NVIDIA": "NVDA",
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
        
    for d in OUTPUT_PARSING_ROOT.iterdir():
        if d.is_dir():
            dir_name = d.name
            
            # Filter logic
            if language == "en":
                if "10k" not in dir_name.lower():
                    continue
                if benchmark_only and dir_name not in BENCHMARK_DIRS:
                    continue

                # Extrapolate Ticker (simple heuristic for US dataset)
                raw_prefix = dir_name.split('_')[0].upper()
                ticker = TICKER_MAP.get(raw_prefix, raw_prefix)
                
                # Fetch fiscal year
                match = re.search(r"202\d", dir_name)
                fiscal_year = int(match.group(0)) if match else 2025

                pages_dir = d / "pages"
                page_count = len(list(pages_dir.glob("page_*.json"))) if pages_dir.exists() else 0

                specs.append({
                    "dir_name": dir_name,
                    "ticker": ticker,
                    "name": dir_name,
                    "pages": page_count,
                    "fiscal_year": fiscal_year,
                })
            elif language == "vi":
                if "10k" in dir_name.lower():
                    continue
                # Simple logic for VN dataset
                ticker = dir_name.split('_')[0].upper()
                match = re.search(r"202\d", dir_name)
                fiscal_year = int(match.group(0)) if match else 2025
                
                specs.append({
                    "dir_name": dir_name,
                    "ticker": ticker,
                    "name": dir_name,
                    "pages": 0,
                    "fiscal_year": fiscal_year,
                })
                
    # Sort for deterministic ordering
    specs.sort(key=lambda x: x["dir_name"])
    return specs

# Default export for scripts
COMPANY_SPECS = get_company_specs(language="en", benchmark_only=False)
BENCHMARK_COMPANY_SPECS = get_company_specs(language="en", benchmark_only=True)
