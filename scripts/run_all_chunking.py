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
    if 1 in active_methods:
        print("\n" + "=" * 70)
        print("▶️ PHƯƠNG PHÁP 1: NAIVE FIXED-SIZE CHUNKING (512 tokens / 100 overlap)")
        print("=" * 70)
        m1_start = time.time()
        for idx, spec in enumerate(target_specs, 1):
            print(f"[{idx}/{len(target_specs)}] M1 xử lý {spec['ticker']} ({spec['dir_name']})...")
            try:
                run_fixed_size_chunking_for_company(spec)
            except Exception as e:
                print(f"   [LỖI] M1 cho {spec['dir_name']}: {e}")
        timings[1] = time.time() - m1_start
        print(f"✅ Hoàn tất Phương pháp 1 trong {timings[1]:.2f}s")

    # --- METHOD 2 ---
    if 2 in active_methods:
        print("\n" + "=" * 70)
        print("▶️ PHƯƠNG PHÁP 2: DETERMINISTIC STRUCTURE-AWARE CHUNKING ($0 LLM, Table-Atomic)")
        print("=" * 70)
        m2_start = time.time()
        for idx, spec in enumerate(target_specs, 1):
            print(f"[{idx}/{len(target_specs)}] M2 xử lý {spec['ticker']} ({spec['dir_name']})...")
            try:
                run_deterministic_chunking_for_company(spec)
            except Exception as e:
                print(f"   [LỖI] M2 cho {spec['dir_name']}: {e}")
        timings[2] = time.time() - m2_start
        print(f"✅ Hoàn tất Phương pháp 2 trong {timings[2]:.2f}s")

    # --- METHOD 3 ---
    if 3 in active_methods:
        print("\n" + "=" * 70)
        print("▶️ PHƯƠNG PHÁP 3: HEADING PRE-SPLIT + LLM MERGE (DeepSeek-Chat, 300-600 words)")
        print("=" * 70)
        m3_start = time.time()
        for idx, spec in enumerate(target_specs, 1):
            print(f"[{idx}/{len(target_specs)}] M3 xử lý {spec['ticker']} ({spec['dir_name']})...")
            try:
                run_method3_for_company(spec)
            except Exception as e:
                print(f"   [LỖI] M3 cho {spec['dir_name']}: {e}")
        timings[3] = time.time() - m3_start
        print(f"✅ Hoàn tất Phương pháp 3 trong {timings[3]:.2f}s")

    # --- METHOD 4 ---
    if 4 in active_methods:
        print("\n" + "=" * 70)
        print("▶️ PHƯƠNG PHÁP 4: LLM BOUNDARY TAGGING (DeepSeek-Chat, Tagged Raw Text)")
        print("=" * 70)
        m4_start = time.time()
        for idx, spec in enumerate(target_specs, 1):
            print(f"[{idx}/{len(target_specs)}] M4 xử lý {spec['ticker']} ({spec['dir_name']})...")
            try:
                run_method4_for_company(spec)
            except Exception as e:
                print(f"   [LỖI] M4 cho {spec['dir_name']}: {e}")
        timings[4] = time.time() - m4_start
        print(f"✅ Hoàn tất Phương pháp 4 trong {timings[4]:.2f}s")

    # --- METHOD 5 ---
    if 5 in active_methods:
        print("\n" + "=" * 70)
        print("▶️ PHƯƠNG PHÁP 5: PROPOSED GOLDEN HYBRID CHUNKING (Dual Representation)")
        print("=" * 70)
        m5_start = time.time()
        for idx, spec in enumerate(target_specs, 1):
            print(f"[{idx}/{len(target_specs)}] M5 xử lý {spec['ticker']} ({spec['dir_name']})...")
            try:
                run_method5_for_company(spec)
            except Exception as e:
                print(f"   [LỖI] M5 cho {spec['dir_name']}: {e}")
        timings[5] = time.time() - m5_start
        print(f"✅ Hoàn tất Phương pháp 5 trong {timings[5]:.2f}s")

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
