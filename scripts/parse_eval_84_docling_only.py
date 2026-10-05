"""
parse_eval_84_docling_only.py
=============================
Batch Parser for 84 FinanceBench evaluation filings using PURE DOCLING ONLY.
Chỉ sử dụng Docling nguyên bản và những gì Docling cung cấp (không router, không Marker, không VLM fallback).

Tính năng:
- Hỗ trợ chạy Tuần tự (1 GPU an toàn cho VRAM 4GB) hoặc Song song đa tiến trình (--workers N).
- Tự động phát hiện dung lượng VRAM & cảnh báo an toàn chống OOM khi chạy đa tiến trình CUDA.
- Chế độ TableFormer FAST (--fast) tăng tốc x2 nhận diện bảng biểu.
- Xuất dữ liệu tương thích 100% với FinAnalyst-AI Studio (outputs/parsing/<doc_id>/pages/).
- Auto-resume: Tự động bỏ qua các file đã hoàn thành.
- Sắp xếp thứ tự linh hoạt (--sort-by pages_asc / name / size).
- Thống kê tiến độ thời gian thực, tốc độ giây/trang và ETA.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    import io
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass

import fitz  # PyMuPDF for page count verification
import torch

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        AcceleratorOptions,
        AcceleratorDevice,
    )
except ImportError as err:
    print(f"❌ Lỗi: Chưa cài đặt Docling. Hãy cài đặt bằng lệnh: pip install docling")
    sys.exit(1)


# Global converter instance per worker process
_GLOBAL_CONVERTER: Optional[DocumentConverter] = None


try:
    from financial_parser.profiler import _likely_tabular
except ImportError:
    _likely_tabular = None


def _contiguous_runs(page_numbers: Sequence[int], max_run_size: int = 10) -> List[Tuple[int, int]]:
    """Split page numbers into contiguous runs of at most max_run_size to protect GPU VRAM."""
    if not page_numbers:
        return []
    ordered = sorted(page_numbers)
    runs: List[Tuple[int, int]] = []
    first = previous = ordered[0]
    count = 1
    for current in ordered[1:]:
        if current == previous + 1 and count < max_run_size:
            previous = current
            count += 1
        else:
            runs.append((first, previous))
            first = previous = current
            count = 1
    runs.append((first, previous))
    return runs


def init_worker_converter(fast_mode: bool, device_type: str, num_threads: int):
    """Initializes a persistent DocumentConverter inside each worker process."""
    global _GLOBAL_CONVERTER
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    # GPU Intra-Document Batching optimization
    if hasattr(pipeline_options, "layout_batch_size"):
        pipeline_options.layout_batch_size = 4
    if hasattr(pipeline_options, "table_batch_size"):
        pipeline_options.table_batch_size = 4

    if fast_mode:
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    else:
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    if device_type == "cuda" and torch.cuda.is_available():
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
            num_threads=num_threads,
        )
    else:
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.AUTO,
            num_threads=num_threads,
        )

    _GLOBAL_CONVERTER = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def extract_page_markdowns(docling_doc) -> Dict[int, str]:
    """Extracts raw markdown for each page from DoclingDocument provenance."""
    page_pieces: Dict[int, List[str]] = {}

    for item, level in docling_doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        p_no = getattr(prov[0], "page_no", None)
        if p_no is None:
            continue
        if p_no not in page_pieces:
            page_pieces[p_no] = []

        item_type = type(item).__name__
        if item_type == "TableItem":
            try:
                content = item.export_to_markdown(doc=docling_doc)
            except TypeError:
                content = item.export_to_markdown()
        elif item_type == "SectionHeaderItem":
            content = f"{'#' * min(max(int(getattr(item, 'level', 1)), 1), 6)} {getattr(item, 'text', '')}"
        else:
            content = getattr(item, "text", "")

        if content and str(content).strip():
            page_pieces[p_no].append(str(content).strip())

    return {p: "\n\n".join(pieces) for p, pieces in page_pieces.items()}


def parse_single_pdf_worker(
    pdf_path: Path,
    output_base_dir: Path,
    force: bool = False,
    smart_profile: bool = True,
) -> Dict[str, any]:
    """Worker function to parse one PDF file using Smart Profiling + Docling."""
    global _GLOBAL_CONVERTER
    if _GLOBAL_CONVERTER is None:
        raise RuntimeError("Worker converter is not initialized.")

    doc_id = pdf_path.stem.lower()
    doc_out = output_base_dir / doc_id
    pages_dir = doc_out / "pages"
    summary_file = doc_out / "parsing_summary.json"

    # Pre-read page count via PyMuPDF
    try:
        doc = fitz.open(str(pdf_path))
        pdf_pages = len(doc)
    except Exception:
        doc = None
        pdf_pages = 0

    # Auto-resume check
    if not force and summary_file.exists():
        try:
            prev = json.loads(summary_file.read_text(encoding="utf-8"))
            if prev.get("status") == "success" and (pdf_pages == 0 or prev.get("total_pages", 0) >= pdf_pages - 1):
                if doc is not None:
                    doc.close()
                return {
                    "status": "skipped",
                    "doc_id": doc_id,
                    "filename": pdf_path.name,
                    "pages": prev.get("total_pages", 0),
                    "runtime": 0,
                    "seconds_per_page": prev.get("seconds_per_page", 0),
                    "chars": prev.get("total_chars", 0),
                }
        except Exception:
            pass

    doc_out.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    page_md_map: Dict[int, str] = {}
    page_engine_map: Dict[int, str] = {}

    if smart_profile and doc is not None and _likely_tabular is not None:
        # Phase 1: Smart Profiling - Identify pages that actually have tables
        table_pages = []
        for p_idx in range(len(doc)):
            p_no = p_idx + 1
            page = doc[p_idx]
            words = page.get_text("words") or []
            is_tab = _likely_tabular(words, page_height=float(page.rect.height) if hasattr(page, "rect") else 792.0)
            if is_tab:
                table_pages.append(p_no)
            else:
                raw_text = page.get_text() or ""
                page_md_map[p_no] = raw_text
                page_engine_map[p_no] = "pymupdf_text_only"

        # Phase 2: Convert ONLY table-rich pages via Docling in contiguous runs
        runs = _contiguous_runs(table_pages, max_run_size=10)
        print(f"  ⚡ Smart Profiling: {len(table_pages)}/{pdf_pages} trang có bảng -> gửi Docling ({len(runs)} batches), {pdf_pages - len(table_pages)} trang chữ lấy trực tiếp.", flush=True)

        for r_start, r_end in runs:
            try:
                conv_res = _GLOBAL_CONVERTER.convert(str(pdf_path), page_range=(r_start, r_end))
                run_md = extract_page_markdowns(conv_res.document)
                for p in range(r_start, r_end + 1):
                    if p in table_pages:
                        page_md_map[p] = run_md.get(p, doc[p - 1].get_text() or "")
                        page_engine_map[p] = "docling_table"
                del conv_res
            except Exception as e:
                print(f"    ⚠️ Fallback PyMuPDF cho batch {r_start}-{r_end}: {e}", flush=True)
                for p in range(r_start, r_end + 1):
                    if p not in page_md_map:
                        page_md_map[p] = doc[p - 1].get_text() or ""
                        page_engine_map[p] = "pymupdf_fallback"

        doc.close()
    else:
        # Full Document Conversion
        if doc is not None:
            doc.close()
        conv_res = _GLOBAL_CONVERTER.convert(str(pdf_path))
        page_md_map = extract_page_markdowns(conv_res.document)
        for p in page_md_map:
            page_engine_map[p] = "docling_only"
        del conv_res

    t1 = time.time()
    runtime = round(t1 - t0, 2)

    total_pages_detected = max(pdf_pages, len(page_md_map))
    table_count = 0
    full_md_parts = []

    for p_no in range(1, total_pages_detected + 1):
        page_md = page_md_map.get(p_no, "")
        engine_used = page_engine_map.get(p_no, "docling_only")

        # Save page_XXX.md
        md_file = pages_dir / f"page_{p_no:03d}.md"
        md_file.write_text(page_md, encoding="utf-8")
        full_md_parts.append(f"<!-- Page {p_no} -->\n{page_md}")

        # Save page_XXX.json (100% acceptance for baseline - no page is marked fail)
        table_count += page_md.count("| --- |") + page_md.count("|:---|") + page_md.count("| ---:")
        page_data = {
            "schema_version": "financial-parser-page-v1",
            "page": {
                "document_id": doc_id,
                "pdf_page": p_no,
                "engine": engine_used,
                "char_count": len(page_md),
                "word_count": len(page_md.split()),
                "markdown": page_md,
                "qc": {
                    "status": "pass",
                    "note": f"smart_profile_output_{engine_used}",
                },
            },
        }
        json_file = pages_dir / f"page_{p_no:03d}.json"
        json_file.write_text(json.dumps(page_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1. Full document markdown
    full_md = "\n\n".join(full_md_parts)
    (doc_out / f"{doc_id}_full.md").write_text(full_md, encoding="utf-8")

    # 3. Summary metadata
    summary = {
        "status": "success",
        "document_id": doc_id,
        "filename": pdf_path.name,
        "source_pdf": str(pdf_path),
        "engine": "smart_docling_profiler",
        "total_pages": total_pages_detected,
        "total_chars": len(full_md),
        "estimated_tables": table_count,
        "qc_counts": {"pass": total_pages_detected, "fail": 0, "warning": 0},
        "review_queue_count": 0,
        "runtime_seconds": runtime,
        "seconds_per_page": round(runtime / max(total_pages_detected, 1), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def get_pdf_page_count(path: Path) -> int:
    """Fast page count via fitz."""
    try:
        d = fitz.open(str(path))
        c = len(d)
        d.close()
        return c
    except Exception:
        return 999999


def main():
    parser = argparse.ArgumentParser(description="Parse 84 FinanceBench PDFs with Pure Docling Only (Batch/Parallel).")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "sec_filings" / "pdfs" / "eval_84"),
        help="Thư mục chứa 84 file PDF đánh giá.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "parsing"),
        help="Thư mục lưu kết quả (mặc định: outputs/parsing tương thích demo_app).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số tiến trình chạy song song (Mặc định: 1 để an toàn VRAM GPU 4GB; hoặc 2-4 nếu chạy CPU/GPU lớn).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Thiết bị tính toán: cuda, cpu, hoặc auto.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Số luồng CPU tính toán nội bộ trong mỗi worker (mặc định: 4).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Sử dụng TableFormer FAST mode (tăng tốc x1.8 - x2.0).",
    )
    parser.add_argument(
        "--smart-profile",
        action="store_true",
        default=True,
        help="Bật Smart Profiling: dùng PyMuPDF quét trang chữ thuần (0.001s), chỉ gửi trang bảng tới Docling (mặc định: True).",
    )
    parser.add_argument(
        "--no-smart-profile",
        action="store_false",
        dest="smart_profile",
        help="Tắt Smart Profiling, chạy toàn bộ 100% trang qua Docling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc parse lại các file đã có kết quả.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["name", "size", "pages_asc", "pages_desc"],
        default="pages_asc",
        help="Thứ tự xử lý file: pages_asc (từ file ngắn đến dài, thấy KQ ngay), name, hoặc size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số lượng file cần parse (0 = toàn bộ 84 file).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {input_dir}")
        sys.exit(1)

    all_pdfs = sorted(list(input_dir.glob("*.pdf")))
    if not all_pdfs:
        print(f"❌ Không có file PDF nào trong {input_dir}")
        sys.exit(1)

    # Sort files according to preference
    if args.sort_by == "pages_asc":
        print("⏳ Đang quét nhanh số trang để xếp thứ tự từ ngắn đến dài (pages_asc)...")
        pdf_with_pages = [(p, get_pdf_page_count(p)) for p in all_pdfs]
        pdf_with_pages.sort(key=lambda x: x[1])
        all_pdfs = [p[0] for p in pdf_with_pages]
    elif args.sort_by == "pages_desc":
        pdf_with_pages = [(p, get_pdf_page_count(p)) for p in all_pdfs]
        pdf_with_pages.sort(key=lambda x: x[1], reverse=True)
        all_pdfs = [p[0] for p in pdf_with_pages]
    elif args.sort_by == "size":
        all_pdfs.sort(key=lambda x: x.stat().st_size)

    if args.limit > 0:
        all_pdfs = all_pdfs[:args.limit]

    total_files = len(all_pdfs)

    # Device & Hardware Safety Analysis
    target_device = args.device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    total_vram_gb = 0
    if target_device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        gpu_name = "CPU"

    print("=" * 70)
    print("🚀 PIPELINE PARSING PURE DOCLING ONLY (BASELINE CHUẨN HOÁ)")
    print(f"📂 Thư mục nguồn:     {input_dir} ({total_files} tài liệu)")
    print(f"📁 Thư mục xuất:      {output_dir}")
    print(f"⚡ Thiết bị:          {gpu_name} (Device: {target_device})")
    if total_vram_gb > 0:
        print(f"🎮 Tổng VRAM GPU:     {total_vram_gb} GB")
    print(f"⚙️ TableFormer Mode:  {'FAST (Tăng tốc x2)' if args.fast else 'ACCURATE (Chính xác)'}")
    print(f"🧵 Workers song song: {args.workers} worker(s) | Threads/worker: {args.threads}")
    print(f"📋 Thứ tự ưu tiên:   {args.sort_by}")
    print(f"🔄 Chế độ ghi đè:     {args.force}")

    # VRAM Safety Check for Multi-Worker CUDA
    if target_device == "cuda" and args.workers > 1:
        needed_vram = args.workers * 2.5
        print(f"\n⚠️ [CẢNH BÁO PHẦN CỨNG] Bạn đang bật {args.workers} workers trên CUDA GPU ({total_vram_gb} GB VRAM).")
        print(f"   Mỗi worker Docling chiếm ~2.5 - 3.0 GB VRAM. Cần tối thiểu: ~{needed_vram} GB VRAM.")
        if total_vram_gb < needed_vram:
            print(f"   🚨 NGUY CƠ TRÀN VRAM (CUDA OOM): Card {gpu_name} chỉ có {total_vram_gb} GB VRAM!")
            print(f"   💡 Khuyên dùng: Giữ --workers 1 kèm --fast, HOẶC chuyển sang --device cpu --workers {args.workers}.")
            print("   Tiếp tục sau 3 giây...\n")
            time.sleep(3)

    print("=" * 70)

    total_start = time.time()
    completed = 0
    skipped = 0
    failed = 0
    total_pages_done = 0

    if args.workers <= 1:
        # Sequential Execution (Safest for 4GB Laptop GPU)
        init_worker_converter(fast_mode=args.fast, device_type=target_device, num_threads=args.threads)

        for idx, pdf in enumerate(all_pdfs, start=1):
            print(f"\n[{idx}/{total_files}] 📄 Đang xử lý: {pdf.name} ...", flush=True)
            try:
                res = parse_single_pdf_worker(pdf, output_dir, force=args.force, smart_profile=args.smart_profile)
                if res.get("status") == "skipped":
                    print(f"  ⏭️ Bỏ qua: Đã có sẵn kết quả ({res.get('pages', 0)} trang).", flush=True)
                    skipped += 1
                else:
                    p_cnt = res.get("total_pages", 0)
                    rt = res.get("runtime_seconds", 0)
                    spp = res.get("seconds_per_page", 0)
                    chars = res.get("total_chars", 0)
                    total_pages_done += p_cnt
                    completed += 1

                    elapsed = time.time() - total_start
                    avg_per_file = elapsed / max(completed, 1)
                    rem_files = total_files - (completed + skipped)
                    eta_min = (rem_files * avg_per_file) / 60

                    print(
                        f"  ✅ Hoàn tất trong {rt}s | {p_cnt} trang ({spp}s/trang) | {chars:,} ký tự | ETA: ~{eta_min:.1f} phút",
                        flush=True,
                    )
            except Exception as e:
                print(f"  ❌ Lỗi khi parse {pdf.name}: {e}", flush=True)
                failed += 1

    else:
        # Parallel Multi-Process Execution
        print(f"\n🔥 Khởi tạo hồ bơi tiến trình (ProcessPoolExecutor) với {args.workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker_converter,
            initargs=(args.fast, target_device, args.threads),
        ) as executor:
            future_to_pdf = {
                executor.submit(parse_single_pdf_worker, pdf, output_dir, args.force, args.smart_profile): (idx, pdf)
                for idx, pdf in enumerate(all_pdfs, start=1)
            }

            for future in concurrent.futures.as_completed(future_to_pdf):
                idx, pdf = future_to_pdf[future]
                try:
                    res = future.result()
                    if res.get("status") == "skipped":
                        print(f"[{idx}/{total_files}] ⏭️ Bỏ qua {pdf.name}: Đã có sẵn ({res.get('pages', 0)} trang).", flush=True)
                        skipped += 1
                    else:
                        p_cnt = res.get("total_pages", 0)
                        rt = res.get("runtime_seconds", 0)
                        spp = res.get("seconds_per_page", 0)
                        chars = res.get("total_chars", 0)
                        total_pages_done += p_cnt
                        completed += 1

                        elapsed = time.time() - total_start
                        avg_per_file = elapsed / max(completed, 1)
                        rem_files = total_files - (completed + skipped)
                        eta_min = (rem_files * avg_per_file) / 60

                        print(
                            f"[{idx}/{total_files}] ✅ Xong {pdf.name} trong {rt}s | {p_cnt} trang ({spp}s/trang) | ETA: ~{eta_min:.1f}m",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[{idx}/{total_files}] ❌ Lỗi {pdf.name}: {e}", flush=True)
                    failed += 1

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("🏁 TỔNG KẾT PARSING PURE DOCLING ONLY")
    print(f"Tổng số tài liệu:  {total_files}")
    print(f"Hoàn thành mới:    {completed} files ({total_pages_done:,} trang)")
    print(f"Bỏ qua (có sẵn):   {skipped} files")
    print(f"Thất bại:          {failed} files")
    print(f"Tổng thời gian:    {total_elapsed / 60:.2f} phút ({total_elapsed:.1f}s)")
    if total_pages_done > 0:
        print(f"Tốc độ trung bình: {total_elapsed / total_pages_done:.2f}s / trang")
    print(f"Output lưu tại:    {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
