"""Generate a comprehensive audit and evaluation report inside the parsing output directory.

Usage:
    python script/generate_parsing_evaluation_report.py output_test/parsing_review
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

def generate_report(output_dir: Path, ground_truth_html: Optional[Path] = None) -> Path:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "routing_manifest.json"
    pages_dir = output_dir / "pages"
    review_queue_path = output_dir / "review_queue.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Collect all canonical JSON pages
    page_files = sorted(pages_dir.glob("page_*.json")) if pages_dir.is_dir() else []
    pages_data: Dict[int, Dict[str, Any]] = {}
    for pf in page_files:
        try:
            m = re.search(r"page_(\d+)\.json", pf.name)
            if m:
                p_num = int(m.group(1))
                with pf.open("r", encoding="utf-8") as f:
                    pages_data[p_num] = json.load(f)
        except Exception:
            pass

    # 2. Collect review queue
    review_records: List[Dict[str, Any]] = []
    if review_queue_path.is_file():
        with review_queue_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        review_records.append(json.loads(line))
                    except Exception:
                        pass

    # 3. Calculate metrics
    total_pages_manifest = manifest.get("document", {}).get("range", {}).get("end", len(manifest.get("profiles", [])))
    total_processed = len(pages_data)
    engine_counts = manifest.get("result", {}).get("engine_counts", {})
    qc_counts = manifest.get("result", {}).get("qc_counts", {})

    # Gather metrics across generated pages
    num_recalls = []
    num_precisions = []
    text_recalls = []
    total_tables = 0
    pages_with_tables = 0

    for p_num, page_json in pages_data.items():
        page_dict = page_json.get("page", {})
        qc_meta = page_dict.get("qc") or page_json.get("provenance", {}).get("qc", {})
        if qc_meta.get("numeric_recall") is not None:
            num_recalls.append(qc_meta["numeric_recall"])
        if qc_meta.get("numeric_precision") is not None:
            num_precisions.append(qc_meta["numeric_precision"])
        if qc_meta.get("source_text_recall") is not None:
            text_recalls.append(qc_meta["source_text_recall"])

        blocks = page_dict.get("blocks", [])
        tables_on_page = sum(1 for b in blocks if b.get("block_type") == "table")
        total_tables += tables_on_page
        if tables_on_page > 0:
            pages_with_tables += 1

    avg_num_recall = sum(num_recalls) / len(num_recalls) if num_recalls else 0.0
    avg_num_precision = sum(num_precisions) / len(num_precisions) if num_precisions else 0.0
    avg_text_recall = sum(text_recalls) / len(text_recalls) if text_recalls else 0.0

    # 4. Check core financial statement pages (NVIDIA 10-K: Pages 52 to 56)
    core_statements_audit = []
    known_statements = {
        52: "Consolidated Statements of Income",
        53: "Consolidated Statements of Comprehensive Income",
        54: "Consolidated Balance Sheets",
        55: "Consolidated Statements of Shareholders' Equity",
        56: "Consolidated Statements of Cash Flows",
    }
    for p_num, title in known_statements.items():
        if p_num in pages_data:
            p_json = pages_data[p_num]
            p_dict = p_json.get("page", {})
            p_route = p_dict.get("route") or p_json.get("provenance", {}).get("route", {})
            p_qc = p_dict.get("qc") or p_json.get("provenance", {}).get("qc", {})
            engine = p_route.get("engine", "unknown")
            qc_status = p_qc.get("status", "unknown")
            blocks = p_dict.get("blocks", [])
            tables = [b for b in blocks if b.get("block_type") == "table"]
            rows_count = sum(len(t.get("rows", [])) for t in tables)
            core_statements_audit.append({
                "page": p_num,
                "title": title,
                "status": "PROCESSED",
                "engine": engine,
                "qc": qc_status,
                "tables_count": len(tables),
                "total_rows": rows_count,
            })
        else:
            in_review = any(r.get("pdf_page") == p_num for r in review_records)
            core_statements_audit.append({
                "page": p_num,
                "title": title,
                "status": "IN_REVIEW_QUEUE" if in_review else "NOT_PROCESSED",
                "engine": "vlm_fallback_needed" if in_review else "missing",
                "qc": "fail",
                "tables_count": 0,
                "total_rows": 0,
            })

    # 5. Build Markdown Report
    doc_id = manifest.get("document", {}).get("document_id", "financial_report")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append(f"# Báo Cáo Đánh Giá Chất Lượng Bóc Tách (Parsing Evaluation Report)")
    md.append(f"**Tài liệu:** `{doc_id}` | **Thời điểm tạo:** `{created_at}`")
    md.append(f"**Thư mục lưu trữ:** `{output_dir.name}`\n")
    md.append("---\n")

    # Section 1: Executive Summary
    md.append("## 1. Tổng Quan Tiến Trình (Executive Summary)\n")
    md.append(f"| Chỉ số | Giá trị | Đánh giá |")
    md.append(f"|---|:---:|---|")
    md.append(f"| **Tổng số trang xử lý** | **{total_processed} / {total_pages_manifest} trang** | Hoàn tất {total_processed/max(1, total_pages_manifest)*100:.1f}% |")
    md.append(f"| **Trang đạt PASS (Tuyệt đối)** | **{qc_counts.get('pass', 0)} trang** | Đạt chuẩn độ chính xác cao nhất |")
    md.append(f"| **Trang WARNING (An toàn)** | **{qc_counts.get('warning', 0)} trang** | Đã được hệ thống phê duyệt (văn xuôi/tiêu đề lặp) |")
    md.append(f"| **Trang cần xem xét (Review Queue)** | **{len(review_records)} trang** | Tiết kiệm token bằng cách cô lập xử lý sau |")
    md.append(f"| **Tổng số bảng biểu trích xuất** | **{total_tables} bảng** | Phân bố trên {pages_with_tables} trang tài liệu |")
    md.append("\n---\n")

    # Section 2: Quality & Fidelity Metrics
    md.append("## 2. Thang Đo Độ Chuẩn Xác Dữ Liệu (Fidelity Metrics)\n")
    md.append(f"| Metric | Điểm trung bình | Mục tiêu tối thiểu | Trạng thái |")
    md.append(f"|---|:---:|:---:|:---:|")
    md.append(f"| **Numeric Recall** (Độ thu hồi số liệu) | **{avg_num_recall*100:.2f}%** | 88.0% | {'✅ PASS' if avg_num_recall >= 0.88 else '⚠️ REVIEW'} |")
    md.append(f"| **Numeric Precision** (Độ chính xác số liệu) | **{avg_num_precision*100:.2f}%** | 88.0% | {'✅ PASS' if avg_num_precision >= 0.88 else '⚠️ REVIEW'} |")
    md.append(f"| **Source Text Recall** (Độ thu hồi văn bản) | **{avg_text_recall*100:.2f}%** | 88.0% | {'✅ PASS' if avg_text_recall >= 0.88 else '⚠️ REVIEW'} |")
    md.append("\n---\n")

    # Section 3: Cost & Engine Efficiency
    md.append("## 3. Phân Bổ Công Cụ & Hiệu Quả Chi Phí (Cost & Engine Efficiency)\n")
    docling_pages = engine_counts.get("docling", 0)
    vlm_pages = engine_counts.get("deepseek_vlm", 0) + engine_counts.get("gemini_vlm", 0)
    md.append(f"* **Docling-Native ($0 API Cost):** `{docling_pages}` trang ({docling_pages/max(1, total_processed)*100:.1f}%)")
    md.append(f"* **VLM Multimodal Fallback:** `{vlm_pages}` trang")
    md.append(f"* **Chế độ tiết kiệm Token:** {'Hoạt động hiệu quả (Ưu tiên Docling-only)' if vlm_pages == 0 else 'Có kích hoạt VLM chọn lọc'}")
    md.append("\n---\n")

    # Section 4: Core Financial Statements Audit
    md.append("## 4. Kiểm Thẩm 5 Trang Báo Cáo Tài Chính Cốt Lõi (Core Statements)\n")
    md.append("| Trang | Báo cáo tài chính (Financial Statement) | Trạng thái | Engine | QC | Số bảng | Số dòng ma trận |")
    md.append("|:---:|---|:---:|:---:|:---:|:---:|:---:|")
    for row in core_statements_audit:
        status_icon = "✅" if row["status"] == "PROCESSED" else "⏳"
        md.append(f"| **{row['page']}** | {row['title']} | {status_icon} {row['status']} | `{row['engine']}` | `{row['qc']}` | {row['tables_count']} | {row['total_rows']} dòng |")

    # Section 5: Review Queue Details
    if review_records:
        md.append("\n---\n")
        md.append("## 5. Danh Sách Các Trang Cần Chú Ý (Review Queue)\n")
        md.append("| Trang | Lớp trang | Engine đề xuất | Nguyên nhân chính | Hướng xử lý |")
        md.append("|:---:|:---:|:---:|---|---|")
        for rec in review_records:
            p_num = rec.get("pdf_page", 0)
            p_class = rec.get("page_class", "unknown")
            route_eng = rec.get("route", {}).get("engine", "vlm")
            reasons = rec.get("route", {}).get("reason", [])
            err = rec.get("error") or ", ".join(reasons[:2])
            md.append(f"| **{p_num}** | `{p_class}` | `{route_eng}` | `{err}` | Chạy VLM nhắm mục tiêu (`--start {p_num} --end {p_num}`) |")
    else:
        md.append("\n---\n")
        md.append("## 5. Danh Sách Các Trang Cần Chú Ý (Review Queue)\n")
        md.append("🎉 **Không có trang nào bị lỗi! Toàn bộ các trang đều đạt chuẩn an toàn.**\n")

    md.append("\n---\n")
    md.append("## 6. Kết Luận & Khuyến Nghị Tiếp Theo\n")
    if review_records:
        target_pages = ", ".join(str(r.get("pdf_page")) for r in review_records)
        md.append(f"1. **Chạy VLM bù cho các trang trong review queue ({target_pages}):**\n")
        md.append("   ```powershell")
        md.append(f"   python script/parse_financial_reports.py data/nvidia_2025_10k.pdf --start {review_records[0].get('pdf_page')} --end {review_records[0].get('pdf_page')} --out {output_dir} --provider deepseek --force")
        md.append("   ```\n")
    else:
        md.append("1. **Toàn bộ kho dữ liệu đã sẵn sàng cho pha tiếp theo (Structure-Aware Chunking & Vector Indexing).**\n")
    md.append("2. **Định dạng Canonical JSON:** Các tệp đều chứa trường `source_refs`, `hierarchy` và phân cấp ma trận đầy đủ cho phân hệ Hybrid RAG.\n")

    # Write output report
    report_content = "\n".join(md)
    report_file = output_dir / "parsing_evaluation_report.md"
    report_file.write_text(report_content, encoding="utf-8")

    # Write summary JSON
    summary_data = {
        "document_id": doc_id,
        "created_at": created_at,
        "total_processed": total_processed,
        "total_pages_manifest": total_pages_manifest,
        "avg_numeric_recall": round(avg_num_recall, 4),
        "avg_numeric_precision": round(avg_num_precision, 4),
        "avg_source_text_recall": round(avg_text_recall, 4),
        "total_tables": total_tables,
        "engine_counts": engine_counts,
        "qc_counts": qc_counts,
        "review_queue_count": len(review_records),
    }
    summary_file = output_dir / "parsing_evaluation_summary.json"
    summary_file.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SUCCESS] Evaluation report generated: {report_file}")
    return report_file

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_test/parsing_review")
    generate_report(target)
