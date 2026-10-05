"""Batch parser for all SEC 10-K filings in data/sec_filings.

Usage:
    python scripts/parse_all_sec_filings.py
    python scripts/parse_all_sec_filings.py --force
    python scripts/parse_all_sec_filings.py --companies apple intel nvidia --years 2024 2025
    python scripts/parse_all_sec_filings.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()
if "LLAMA_INDEX_API_KEY" in os.environ and not os.environ.get("LLAMA_CLOUD_API_KEY"):
    os.environ["LLAMA_CLOUD_API_KEY"] = os.environ["LLAMA_INDEX_API_KEY"]

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = _ROOT / "scripts"
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass

from finanalyst.paths import DATA_DIR, OUTPUT_PARSING
from financial_parser import FinancialReportRouter, ParserConfig
from financial_parser.engines import EngineUnavailableError


def find_sec_filings(
    base_dir: Path,
    companies: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> List[Path]:
    """Find all SEC 10-K PDF filings matching the given companies and years."""
    pdf_files: List[Path] = []
    if not base_dir.exists():
        return pdf_files

    for pdf in sorted(base_dir.rglob("*.pdf")):
        parts = [p.lower() for p in pdf.parts]
        
        # Check company filter
        if companies:
            if not any(c.lower() in parts or c.lower() in pdf.stem.lower() for c in companies):
                continue
                
        # Check year filter
        if years:
            year_match = False
            for y in years:
                if str(y) in parts or str(y) in pdf.stem:
                    year_match = True
                    break
            if not year_match:
                continue

        pdf_files.append(pdf)
    return pdf_files


def get_doc_output_name(pdf_path: Path) -> str:
    """Generate canonical directory name: e.g., bestbuy_2024q2_10q or amazon_2021_10k."""
    return pdf_path.stem.lower()


def parse_all_filings(
    pdf_files: List[Path],
    output_root: Path,
    config: ParserConfig,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Run parser on all PDF files sequentially."""
    total = len(pdf_files)
    print(f"\n=======================================================", flush=True)
    print(f"🚀 BẮT ĐẦU PIPELINE PARSING: {total} TÀI LIỆU SEC", flush=True)
    print(f"📁 Thư mục output: {output_root}", flush=True)
    print(f"🤖 VLM Provider: {config.vlm_provider} | Model: {config.openrouter_model if config.vlm_provider == 'openrouter' else (config.groq_model if config.vlm_provider == 'groq' else 'default')}", flush=True)
    print(f"⚙️ Chế độ: force={force}, dry_run={dry_run}", flush=True)
    print(f"=======================================================\n", flush=True)

    router = FinancialReportRouter(config)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        doc_name = get_doc_output_name(pdf_path)
        doc_out = output_root / doc_name
        manifest_file = doc_out / "routing_manifest.json"

        pages_dir = doc_out / "pages"
        is_completed = pages_dir.exists() and any(pages_dir.glob("page_*.json"))
        
        if is_completed and not force:
            print(f"  ⏭️ [{idx}/{total}] Đã tồn tại kết quả hoàn chỉnh tại {doc_out}. Bỏ qua (dùng --force để parse lại).", flush=True)
            skipped += 1
            continue

        print(f"  📄 [{idx}/{total}] Đang xử lý: {pdf_path.name} ...", flush=True)
        start_time = time.time()
        try:
            summary = router.process(
                pdf_path,
                output_dir=doc_out,
                force=force,
                dry_run=dry_run,
            )
            elapsed = time.time() - start_time
            if dry_run:
                pages_cnt = summary.get("pages_profiled", 0)
                classes = summary.get("classes", {})
                print(f"  🔍 [DRY-RUN] Đã quét {pages_cnt} trang trong {elapsed:.1f}s | Phân loại: {classes}", flush=True)
            else:
                processed_map = summary.get("processed", {})
                pages_cnt = len(processed_map)
                engines = summary.get("engine_counts", {})
                qc_summary = summary.get("qc_counts", {})
                review_cnt = summary.get("review_queue_count", 0)
                print(f"  ✅ Hoàn tất {pdf_path.name} trong {elapsed:.1f}s | "
                      f"Pages: {pages_cnt} | "
                      f"Engines: {engines} | "
                      f"QC: {qc_summary} | "
                      f"Review Queue: {review_cnt}", flush=True)
            
            # Generate markdown report
            if not dry_run:
                try:
                    from generate_parsing_evaluation_report import generate_report
                    generate_report(doc_out)
                except Exception as rep_err:
                    print(f"  ⚠️ Không thể tạo báo cáo MD: {rep_err}", flush=True)

            completed += 1
        except Exception as err:
            elapsed = time.time() - start_time
            print(f"  ❌ Lỗi khi xử lý {pdf_path.name}: {err}", flush=True)
            failed += 1
        finally:
            gc.collect()

    print(f"\n=======================================================", flush=True)
    print(f"🏁 TỔNG KẾT PIPELINE PARSING", flush=True)
    print(f"Tổng số file: {total} | Hoàn tất: {completed} | Đã có sẵn: {skipped} | Lỗi: {failed}", flush=True)
    print(f"=======================================================\n", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse SEC Form 10-K / 10-Q filings into canonical JSON.")
    parser.add_argument("--file", type=Path, default=None, help="Parse a single specific PDF file directly.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "sec_filings", help="Base directory containing SEC filings.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PARSING, help="Output root directory.")
    parser.add_argument("--companies", nargs="+", default=None, help="Filter specific companies (e.g. apple intel nvda).")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Filter specific years (e.g. 2024 2025).")
    parser.add_argument("--provider", choices=["openrouter", "groq", "deepseek", "gemini", "llamaparse", "auto"], default="openrouter", help="VLM provider.")
    parser.add_argument("--model", default=None, help="VLM model override (default: stealth/space-bunny-alpha for openrouter, qwen/qwen3.8-27b for groq).")
    parser.add_argument("--force", action="store_true", help="Reprocess even if canonical JSON exists.")
    parser.add_argument("--dry-run", action="store_true", help="Profile and manifest only; no heavy extraction.")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM calls.")
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Enable saving intermediate page PNG images for review/flawed pages (default: False to save disk space).",
    )
    parser.add_argument(
        "--no-images",
        dest="save_images",
        action="store_false",
        help="Explicitly disable saving intermediate page PNG images (default behavior).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ParserConfig.from_environment(
        output_root=args.out,
        vlm_provider=args.provider,
        vlm_model=args.model,
        save_intermediate_images=args.save_images,
    )
    if args.no_vlm:
        config = replace(
            config,
            vlm_provider="deepseek",
            llamaparse_api_key="",
            deepseek_api_key="",
            gemini_api_key="",
            groq_api_key="",
            openrouter_api_key="",
        )

    if args.file:
        file_path = args.file.resolve()
        if not file_path.exists():
            print(f"❌ File không tồn tại: {file_path}")
            return 1
        pdf_files = [file_path]
    else:
        pdf_files = find_sec_filings(args.data_dir, companies=args.companies, years=args.years)
        
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào với bộ lọc đã chọn.")
        return 0

    parse_all_filings(
        pdf_files=pdf_files,
        output_root=args.out,
        config=config,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
