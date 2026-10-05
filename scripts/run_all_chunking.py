"""
run_all_chunking.py

Script điều phối thực thi 5 chiến lược phân đoạn tài liệu tài chính (Financial Chunker):
1. Method 1: Naive Fixed-Size (512 tokens / 100 overlap)
2. Method 2: Deterministic Structure-Aware ($0 LLM, Table-Atomic)
3. Method 3: Heading Pre-Split + Batch LLM Merge (DeepSeek-Chat, 300-600 words)
4. Method 4: LLM Semantic Boundary Tagging (DeepSeek-Chat, split-point tagging)
5. Method 5: Proposed Golden Hybrid (Dual Representation: Tuples + 2D Tables, Table-Atomic, Footnote Binding)

Sau khi hoàn tất, script tự động sinh MASTER_CHUNKING_COMPARISON_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from financial_chunker.config import (
    BENCHMARK_COMPANY_SPECS,
    COMPANY_SPECS,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_PARSING_ROOT,
    get_company_specs,
)
from financial_chunker.generate_master_comparison_report import generate_master_report
from financial_chunker.method1_fixed_size import run_fixed_size_chunking_for_company
from financial_chunker.method2_deterministic import run_deterministic_chunking_for_company
from financial_chunker.method3_heading_llm import run_method3_for_company
from financial_chunker.method4_boundary_tagging import run_method4_for_company
from financial_chunker.method5_proposed_golden_hybrid import run_method5_for_company


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy 5 chiến lược chunking tài liệu tài chính.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Chỉ chạy 7 hồ sơ cốt lõi Benchmark (mặc định: chạy TOÀN BỘ 35 hồ sơ SEC 10-K).",
    )
    parser.add_argument(
        "--companies",
        type=str,
        default="",
        help="Danh sách thư mục công ty cần chạy (ngăn cách bởi dấu phẩy, vd: amazon_2024_10k,amd_2025_10k).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="1,2,3,4,5",
        help="Các phương pháp cần chạy (vd: 1,2 hoặc 1,2,3,4,5). Mặc định: 1,2,3,4,5.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số luồng chạy song song (cho local vLLM Continuous Batching). Mặc định: 4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Xác định tập hồ sơ xử lý (Mặc định: Toàn bộ 35 filings)
    if args.companies:
        selected_dirs = [c.strip() for c in args.companies.split(",") if c.strip()]
        target_specs = [s for s in COMPANY_SPECS if s["dir_name"] in selected_dirs]
    elif args.benchmark:
        target_specs = BENCHMARK_COMPANY_SPECS
    else:
        target_specs = COMPANY_SPECS

    active_methods = [int(m.strip()) for m in args.methods.split(",") if m.strip().isdigit()]

    print("================================================================================")
    print("🚀 BẮT ĐẦU CHẠY PIPELINE PHÂN ĐOẠN TÀI LIỆU TÀI CHÍNH (FINANCIAL CHUNKER)")
    print(f"   Thời gian: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   Số lượng hồ sơ: {len(target_specs)}")
    print(f"   Danh sách hồ sơ: {[s['dir_name'] for s in target_specs]}")
    print(f"   Phương pháp kích hoạt: {active_methods}")
    print("================================================================================\n")

    overall_start = time.time()
    timings: Dict[int, float] = {}

    # --- METHOD 1 ---
    def execute_method(method_id: int, method_title: str, method_fn) -> float:
        print("\n" + "=" * 70)
        print(f"▶️ PHƯƠNG PHÁP {method_id}: {method_title}")
        print(f"   (Xử lý {len(target_specs)} công ty | workers={args.workers})")
        print("=" * 70)
        m_start = time.time()
        if args.workers <= 1 or len(target_specs) <= 1:
            for idx, spec in enumerate(target_specs, 1):
                print(f"[{idx}/{len(target_specs)}] M{method_id} xử lý {spec['ticker']} ({spec['dir_name']})...")
                try:
                    method_fn(spec)
                except Exception as e:
                    print(f"   [LỖI] M{method_id} cho {spec['dir_name']}: {e}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            completed_cnt = 0
            total_cnt = len(target_specs)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(method_fn, spec): spec for spec in target_specs}
                for fut in as_completed(futures):
                    spec = futures[fut]
                    completed_cnt += 1
                    try:
                        fut.result()
                        print(f"   ✅ [{completed_cnt}/{total_cnt}] M{method_id} hoàn tất: {spec['ticker']} ({spec['dir_name']})", flush=True)
                    except Exception as e:
                        print(f"   ❌ [{completed_cnt}/{total_cnt}] M{method_id} lỗi: {spec['dir_name']} -> {e}", flush=True)

        elapsed = time.time() - m_start
        print(f"✅ Hoàn tất Phương pháp {method_id} trong {elapsed:.2f}s (~{elapsed/60:.1f} phút)")
        return elapsed

    # --- METHOD 1 ---
    if 1 in active_methods:
        timings[1] = execute_method(1, "NAIVE FIXED-SIZE CHUNKING (512 tokens / 100 overlap)", run_fixed_size_chunking_for_company)

    # --- METHOD 2 ---
    if 2 in active_methods:
        timings[2] = execute_method(2, "DETERMINISTIC STRUCTURE-AWARE CHUNKING ($0 LLM, Table-Atomic)", run_deterministic_chunking_for_company)

    # --- METHOD 3 ---
    if 3 in active_methods:
        timings[3] = execute_method(3, "HEADING PRE-SPLIT + LLM MERGE (300-600 words)", run_method3_for_company)

    # --- METHOD 4 ---
    if 4 in active_methods:
        timings[4] = execute_method(4, "LLM BOUNDARY TAGGING (Tagged Raw Text)", run_method4_for_company)

    # --- METHOD 5 ---
    if 5 in active_methods:
        timings[5] = execute_method(5, "PROPOSED GOLDEN HYBRID CHUNKING (Dual Representation)", run_method5_for_company)

    # --- TỔNG HỢP MASTER REPORT ---
    print("\n" + "=" * 70)
    print("📊 ĐANG SINH BÁO CÁO MASTER COMPARISON REPORT...")
    print("=" * 70)
    try:
        report_content = generate_master_report(specs=target_specs)
        report_path = OUTPUT_CHUNKING_ROOT / "MASTER_CHUNKING_COMPARISON_REPORT.md"
        print(f"✅ Đã ghi Master Comparison Report tại: {report_path}")
    except Exception as e:
        print(f"⚠️ Lỗi sinh Master Report: {e}")

    total_duration = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH TRONG {total_duration:.2f}s")
    for m, t in sorted(timings.items()):
        print(f"   - Method {m}: {t:.2f}s")
    print("================================================================================")


if __name__ == "__main__":
    main()
