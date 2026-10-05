"""
FinanceBench Exact Baseline Parser (PyMuPDF)
============================================
Replicates 100% faithfully the exact PDF text extraction method used by the
original FinanceBench paper authors (Patronus AI & Stanford University):

    from langchain.document_loaders import PyMuPDFLoader
    pdf_reader = PyMuPDFLoader(path_doc)
    pdf_text = pdf_reader.load()

Under the hood, PyMuPDFLoader calls fitz.open(path) and page.get_text() for
each page without layout analysis, table detection, or OCR.

Output Directory:
    outputs/parsing_pymupdf/<doc_id>/
        ├── pages/
        │   ├── page_001.json
        │   ├── page_001.txt
        │   └── ...
        ├── <doc_id>_full.txt
        └── parsing_summary.json

Features:
- Pure CPU execution (no GPU required, 0 VRAM usage)
- Multiprocessing support for ultra-fast extraction (~1-2 minutes for all 84 PDFs)
- Compatible with downstream chunking and vector indexing pipelines
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "sec_filings" / "pdfs" / "eval_84"
OUT_BASE_DIR = PROJECT_ROOT / "outputs" / "parsing_pymupdf"


def parse_single_pdf_pymupdf(pdf_path: Path, force: bool = False) -> Dict[str, Any]:
    """Parse a single PDF using PyMuPDF raw text extraction matching the paper."""
    doc_id = pdf_path.stem.lower()
    doc_out_dir = OUT_BASE_DIR / doc_id
    pages_dir = doc_out_dir / "pages"
    summary_file = doc_out_dir / "parsing_summary.json"
    full_text_file = doc_out_dir / f"{doc_id}_full.txt"

    if summary_file.exists() and not force:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "skipped", "doc_id": doc_id, "pages": data.get("total_pages", 0), "duration": 0}
        except Exception:
            pass

    pages_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        full_text_parts = []
        total_chars = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            raw_text = page.get_text() or ""
            total_chars += len(raw_text)

            # 1. Save page json schema
            page_json_path = pages_dir / f"page_{page_num:03d}.json"
            page_data = {
                "schema_version": "financial-parser-page-v1",
                "page": {
                    "document_id": doc_id,
                    "pdf_page": page_num,
                    "engine": "pymupdf_paper_baseline",
                    "char_count": len(raw_text),
                    "word_count": len(raw_text.split()),
                    "markdown": raw_text,  # PyMuPDF raw text used as content
                    "qc": {
                        "status": "pass",
                        "note": "pymupdf_paper_baseline_raw_text"
                    }
                }
            }
            with open(page_json_path, "w", encoding="utf-8") as pf:
                json.dump(page_data, pf, ensure_ascii=False, indent=2)

            # 2. Save page raw txt
            page_txt_path = pages_dir / f"page_{page_num:03d}.txt"
            with open(page_txt_path, "w", encoding="utf-8") as tf:
                tf.write(raw_text)

            full_text_parts.append(f"--- Page {page_num} ---\n{raw_text}")

        doc.close()

        # Save merged full text
        with open(full_text_file, "w", encoding="utf-8") as ff:
            ff.write("\n\n".join(full_text_parts))

        duration = time.time() - t0
        summary = {
            "document_id": doc_id,
            "engine": "pymupdf_paper_baseline",
            "pdf_source": str(pdf_path.name),
            "total_pages": total_pages,
            "total_characters": total_chars,
            "duration_seconds": round(duration, 2),
            "pages_per_second": round(total_pages / max(0.01, duration), 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(summary_file, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)

        return {
            "status": "success",
            "doc_id": doc_id,
            "pages": total_pages,
            "chars": total_chars,
            "duration": duration
        }

    except Exception as e:
        return {
            "status": "error",
            "doc_id": doc_id,
            "error": str(e),
            "duration": time.time() - t0
        }


def main():
    parser = argparse.ArgumentParser(description="FinanceBench Exact PyMuPDF Baseline Parser")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU worker processes")
    parser.add_argument("--doc_id", type=str, default=None, help="Process a single document")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parsed files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    args = parser.parse_args()

    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.doc_id:
        target_pdfs = list(PDF_DIR.glob(f"*{args.doc_id}*.pdf"))
        if not target_pdfs:
            print(f"❌ PDF for {args.doc_id} not found in {PDF_DIR}")
            return
        res = parse_single_pdf_pymupdf(target_pdfs[0], force=args.force)
        print(f"[{res['status'].upper()}] {res['doc_id']}: {res.get('pages', 0)} pages in {res.get('duration', 0):.2f}s")
        return

    all_pdfs = sorted(list(PDF_DIR.glob("*.pdf")))
    if args.limit:
        all_pdfs = all_pdfs[:args.limit]

    print("\n" + "=" * 70)
    print("🚀 FINANCEBENCH EXACT PYMUPDF BASELINE PARSER")
    print("=" * 70)
    print(f"📁 Input Directory: {PDF_DIR} ({len(all_pdfs)} files)")
    print(f"📂 Output Directory: {OUT_BASE_DIR}")
    print(f"⚙️  Workers: {args.workers} CPU cores | Force overwrite: {args.force}\n")

    t_start = time.time()
    total_pages = 0
    success_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(parse_single_pdf_pymupdf, p, args.force): p for p in all_pdfs}

        for idx, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                res = fut.result()
                status = res["status"].upper()
                pages = res.get("pages", 0)
                dur = res.get("duration", 0)

                if status == "SUCCESS":
                    total_pages += pages
                    success_count += 1
                    print(f"[{idx:02d}/{len(all_pdfs):02d}] ✅ {res['doc_id']:<35} | {pages:>4} pages | {dur:>5.2f}s")
                elif status == "SKIPPED":
                    total_pages += pages
                    success_count += 1
                    print(f"[{idx:02d}/{len(all_pdfs):02d}] ⏭️  {res['doc_id']:<35} | {pages:>4} pages (Skipped)")
                else:
                    print(f"[{idx:02d}/{len(all_pdfs):02d}] ❌ {res['doc_id']:<35} | Error: {res.get('error')}")

            except Exception as e:
                print(f"[{idx:02d}/{len(all_pdfs):02d}] ❌ Error processing {p.name}: {e}")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("🎉 PyMuPDF Baseline Extraction Complete!")
    print(f"📊 Completed: {success_count}/{len(all_pdfs)} PDFs")
    print(f"📄 Total Pages: {total_pages:,} pages")
    print(f"⏱️  Total Duration: {total_time:.2f}s ({total_pages / max(0.1, total_time):.1f} pages/sec)")
    print(f"📂 Output Path: {OUT_BASE_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
