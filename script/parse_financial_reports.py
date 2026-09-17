"""CLI entrypoint for the maintained financial-report parser.

Examples:
    python script/parse_financial_reports.py data/MWG_BCTC_2025.pdf --out output/mwg
    python script/parse_financial_reports.py data/NVIDIA-2025-Annual-Report.pdf --start 50 --end 60
    python script/parse_financial_reports.py data/MWG_BCTC_2025.pdf --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from financial_parser import FinancialReportRouter, ParserConfig
from financial_parser.engines import EngineUnavailableError


if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SEC Form 10-K, IFRS and VAS financial PDFs into auditable canonical page JSON."
    )
    parser.add_argument("pdf_path", type=Path, help="Input PDF, for example an SEC 10-K or BCTC.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory; defaults to output_financial_parser/<PDF name>.")
    parser.add_argument("--start", type=int, default=1, help="First PDF page (one-indexed).")
    parser.add_argument("--end", type=int, default=None, help="Last PDF page (inclusive).")
    parser.add_argument("--provider", choices=["deepseek", "gemini"], default=None, help="VLM provider for vision/scanned/fallback routes (default: deepseek).")
    parser.add_argument("--model", default=None, help="VLM model name (e.g., deepseek-flash or gemini-flash-latest); overrides env.")
    parser.add_argument("--dpi", type=int, default=None, help="Raster DPI for VLM-routed pages; overrides FINANCIAL_PARSER_DPI.")
    parser.add_argument("--force", action="store_true", help="Reprocess pages even when canonical JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write profile and routes only; do not call Docling or VLM.")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM calls; pages requiring VLM enter review_queue.jsonl.")
    parser.add_argument("--fail-on-review", action="store_true", help="Return exit code 2 when any page enters review queue.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ParserConfig.from_environment(
        output_root=args.out,
        vlm_provider=args.provider,
        vlm_model=args.model,
        render_dpi=args.dpi,
    )
    if args.no_vlm:
        config = replace(config, deepseek_api_key="", gemini_api_key="")
    router = FinancialReportRouter(config)

    try:
        summary = router.process(
            args.pdf_path,
            start_page=args.start,
            end_page=args.end,
            output_dir=args.out,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, EngineUnavailableError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_review and summary.get("review_queue_count", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
