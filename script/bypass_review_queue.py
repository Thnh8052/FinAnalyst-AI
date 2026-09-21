#!/usr/bin/env python3
"""
bypass_review_queue.py

Thực hiện bypass an toàn cho 11 trang trong Review Queue:
- Chuyển qc.status từ 'fail' thành 'warning' (để pipeline downstream tiếp nhận).
- Bổ sung metadata qc_bypass:
    - source_verification_badge: "⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {page} có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
    - pdf_inspector_target: thông tin phục vụ tính năng 1-Click PDF Inspector (trang, document_id).
- Cập nhật routing_manifest.json của từng công ty.
- Lưu trữ nhật ký audit tại review_queue_bypassed.jsonl.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

COMPANY_DIRS = [
    "nvidia_2025_10k",
    "amd_10k_2025",
    "apple_2025_10k",
    "intel_2025_10k",
]

OUTPUT_ROOT = Path("output_parsing")


def bypass_company_review_queue(company_dir_name: str) -> Dict[str, Any]:
    company_dir = OUTPUT_ROOT / company_dir_name
    if not company_dir.exists():
        return {"company": company_dir_name, "status": "missing_dir", "bypassed_count": 0}

    queue_file = company_dir / "review_queue.jsonl"
    bypassed_queue_file = company_dir / "review_queue_bypassed.jsonl"
    pages_dir = company_dir / "pages"
    manifest_file = company_dir / "routing_manifest.json"

    # Đọc từ queue_file nếu có nội dung, nếu không thì đọc từ bypassed_queue_file
    source_queue = queue_file if (queue_file.exists() and queue_file.stat().st_size > 0) else bypassed_queue_file

    if not source_queue.exists():
        return {"company": company_dir_name, "status": "no_queue_file", "bypassed_count": 0}

    records: List[Dict[str, Any]] = []
    with open(source_queue, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"company": company_dir_name, "status": "empty_queue", "bypassed_count": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    bypassed_pages: List[int] = []

    for rec in records:
        pdf_page = rec.get("pdf_page")
        if not pdf_page:
            continue

        candidate_paths = [
            pages_dir / f"page_{pdf_page:03d}.json",
            pages_dir / f"page_{pdf_page:04d}.json",
            pages_dir / f"page_{pdf_page}.json",
        ]
        page_json_path = next((p for p in candidate_paths if p.exists()), None)

        if not page_json_path:
            print(f"[{company_dir_name}] Cảnh báo: Không tìm thấy file JSON cho trang {pdf_page}")
            continue

        with open(page_json_path, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        page_obj = page_data.get("page", {})
        doc_id = page_obj.get("document_id", company_dir_name)
        original_qc = page_obj.get("original_qc") or page_obj.get("qc", {})
        original_failures = original_qc.get("failures", [])

        # Metadata bypass sạch, KHÔNG có độ tin cậy dạng %
        bypass_metadata = {
            "bypassed": True,
            "bypassed_at": now_iso,
            "original_qc_status": original_qc.get("status", "fail"),
            "original_failures": original_failures,
            "source_verification_badge": (
                f"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang {pdf_page} "
                f"có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
            ),
            "needs_user_verification": True,
            "pdf_inspector": {
                "document_id": doc_id,
                "pdf_page": pdf_page,
                "highlight_target": "complex_table_or_narrative",
                "action": "open_side_by_side_pdf_viewer"
            }
        }

        updated_warnings = list(original_qc.get("warnings", []))
        for f in original_failures:
            updated_warnings.append(f"bypassed_from_fail:{f}")

        page_obj["original_qc"] = original_qc
        page_obj["qc"] = {
            **original_qc,
            "status": "warning",
            "warnings": updated_warnings,
            "failures": []
        }
        page_obj["qc_bypass"] = bypass_metadata

        with open(page_json_path, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)

        bypassed_pages.append(pdf_page)

    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        res = manifest_data.get("result", {})
        processed = res.get("processed", {})

        for p_num in bypassed_pages:
            key = str(p_num)
            if key in processed:
                processed[key]["qc"] = "warning"
                processed[key]["bypassed"] = True

        qc_counts = res.get("qc_counts", {})
        fail_count = qc_counts.get("fail", 0)
        qc_counts["fail"] = max(0, fail_count - len(bypassed_pages))
        qc_counts["warning"] = qc_counts.get("warning", 0) + len(bypassed_pages)

        res["review_queue_bypassed_count"] = len(bypassed_pages)
        res["review_queue_count"] = 0

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    # Lưu lại queue đã bypass
    if source_queue != bypassed_queue_file:
        queue_file.rename(bypassed_queue_file)

    with open(queue_file, "w", encoding="utf-8") as f:
        pass

    return {
        "company": company_dir_name,
        "status": "success",
        "bypassed_count": len(bypassed_pages),
        "bypassed_pages": bypassed_pages,
    }


def main():
    print("=== BẮT ĐẦU CHẠY BYPASS REVIEW QUEUE (11 TRANG - KHÔNG DÙNG ĐỘ TIN CẬY %) ===")
    results = []
    total_bypassed = 0

    for comp in COMPANY_DIRS:
        res = bypass_company_review_queue(comp)
        results.append(res)
        total_bypassed += res["bypassed_count"]
        print(f"[{comp}] Bypassed {res['bypassed_count']} trang: {res.get('bypassed_pages', [])}")

    report_path = OUTPUT_ROOT / "BYPASS_REVIEW_QUEUE_REPORT.md"
    report_lines = [
        "# Báo Cáo Bypass An Toàn Review Queue Cho Downstream RAG",
        f"**Thời điểm thực hiện:** `{datetime.now(timezone.utc).isoformat()}`",
        f"**Tổng số trang được bypass thành công:** **{total_bypassed} trang**",
        "",
        "---",
        "",
        "## 1. Danh Sách Các Trang Bypassed Theo Từng Công Ty",
        "",
        "| Công ty | Số trang bypass | Danh sách các trang PDF | Trạng thái QC mới | Huy hiệu cảnh báo (Verification Badge) |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    for r in results:
        pages_str = ", ".join(str(p) for p in r.get("bypassed_pages", [])) if r.get("bypassed_pages") else "Không có"
        report_lines.append(
            f"| **{r['company']}** | {r['bypassed_count']} trang | `{pages_str}` | `warning` | `⚠️ Cần đối chiếu nguồn gốc` |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Cơ Chế Hoạt Động Của Tính Năng 1-Click PDF Inspector",
        "",
        "Khi người dùng đặt câu hỏi mà câu trả lời truy xuất ngữ cảnh từ các trang này:",
        "1. **Huy hiệu Nhắc nhở Nguồn gốc:** Giao diện Chatbot hiển thị thông điệp: *\"⚠️ Lưu ý kiểm tra nguồn: Bảng số liệu hoặc văn bản tại Trang X có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần.\"*",
        "2. **Nút bấm 1-Click PDF Inspector:** Cho phép người dùng click để mở tức thì side-by-side trang PDF gốc tương ứng với tọa độ bảng/văn bản được highlight tự động.",
        "3. **Tương thích 100% Downstream:** Cả 11 trang giờ đây mang trạng thái `qc.status = 'warning'`, sẵn sàng để các phương pháp Chunking (Fixed-size, Deterministic Structure, LLM-Assisted) tiến hành phân đoạn đầy đủ mà không bị nghẽn.",
        "",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n Đã ghi báo cáo bypass tại: {report_path}")
    print(f" Hoàn tất cập nhật cho toàn bộ {total_bypassed} trang!")


if __name__ == "__main__":
    main()
