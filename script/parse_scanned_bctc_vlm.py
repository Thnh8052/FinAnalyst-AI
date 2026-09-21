"""
Pipeline Trích xuất Toàn bộ Báo cáo Tài chính dạng Scanned PDF bằng Vision-Language Model
- Nguồn dữ liệu: data/vietnamese_bctc/mwg_2025.pdf (46 trang)
- Xử lý: Multi-threading song song qua OpenRouter Free API (ling-3.0-flash-vl:free / qwen3.8-27b:free)
- Cơ chế: Auto-Orientation, Caching từng trang (resume khi lỗi mạng), Exponential Retry on 429
- Đầu ra: output_parsing/mwg_2025/ (pages/*.md, pages/*.json, mwg_2025_full.md)
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

DEFAULT_PDF = "data/vietnamese_bctc/mwg_2025.pdf"
DEFAULT_OUT_DIR = "output_parsing/mwg_2025"

FREE_MODELS = [
    "inclusionai/ling-3.0-flash-vl:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3.8-27b:free",
]

PROMPT = """Bạn là chuyên gia phân tích tài chính và OCR tài liệu quét chuyên nghiệp.
Nhiệm vụ: Đọc toàn bộ nội dung trong trang hình ảnh Báo Cáo Tài Chính quét này và chuyển đổi thành định dạng Markdown chuẩn xác 100%.

Yêu cầu nghiêm ngặt:
1. KHÔI PHỤC DẤU TIẾNG VIỆT CHUẨN XÁC:
   - Dựa vào ngữ cảnh thuật ngữ kế toán tài chính và ngữ pháp tiếng Việt, hãy tự động phục hồi các lỗi mất dấu, thiếu dấu hoặc mờ nét do bản scan tạo ra (ví dụ: 'cong ty' -> 'Công ty', 'ngan han' -> 'ngắn hạn', 'nam giu' -> 'nắm giữ', 'khau hao' -> 'khấu hao', 'du phong' -> 'dự phòng').
   - Tuyệt đối giữ nguyên vẹn tiếng Việt có dấu, không tự ý dịch sang tiếng Anh/Pháp và không nhận diện nhầm sang các ký tự lạ.
2. CẤU TRÚC BẢNG BIỂU TÀI CHÍNH:
   - Chuyển thành bảng Markdown với đầy đủ các cột: | Mã số | TÀI SẢN / CHỈ TIÊU | Thuyết minh | Số cuối năm / Kỳ này | Số đầu năm / Kỳ trước |
   - Tuyệt đối không được bỏ sót cột Thuyết minh (nếu có số ghi chú 5, 6, 7... phải điền đầy đủ).
   - Trích xuất đầy đủ tất cả các dòng, bao gồm Mã số con (111, 112, 123, 131, 132, 216, 221, 222...).
3. ĐỘ CHÍNH XÁC SỐ LIỆU KẾ TOÁN:
   - Giữ nguyên số kế toán từng chữ số, không làm tròn, giữ nguyên dấu chấm phân cách hàng nghìn (ví dụ: 77.201.650.782.966).
   - Các số âm trong ngoặc đơn kế toán ví dụ (609.594.155.009) phải giữ nguyên định dạng ngoặc đơn.
4. PHẦN THUYẾT MINH VĂN BẢN (NARRATIVE):
   - Giữ đúng cấu trúc phân cấp tiêu đề (#, ##, ###), danh sách gạch đầu dòng và đoạn văn.
5. ĐỊNH DẠNG ĐẦU RA:
   - Chỉ xuất ra nội dung Markdown của trang, không thêm lời chào, bình luận hay mở đầu/kết thúc rườm rà.
"""

def render_page_to_png(doc: fitz.Document, page_idx: int, dpi: int = 140) -> bytes:
    """Render 1 trang PDF ra ảnh PNG bytes, tự động xoay nếu trang nằm ngang."""
    page = doc[page_idx]
    rect = page.rect
    
    # Kiểm tra hướng trang: Nếu Chiều ngang > Chiều dọc -> Trang ngang (Landscape)
    # Cần xoay 90 độ để mô hình đọc chữ xuôi chiều
    rotation = page.rotation
    if rect.width > rect.height and rotation == 0:
        rotation = 90
        
    mat = fitz.Matrix(dpi / 72, dpi / 72).prerotate(rotation)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def call_openrouter_with_retry(img_b64: str, api_key: str, models_list: list = None, max_retries: int = 4) -> str:
    """Gọi OpenRouter API với cơ chế Exponential Backoff khi gặp Rate Limit 429."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FinAnalyst-AI",
        "X-Title": "FinAnalyst-AI"
    }

    models_to_try = models_list if models_list else FREE_MODELS

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                
                elif resp.status_code == 429:
                    wait_sec = (2 ** attempt) + 2  # 3s, 4s, 6s, 10s...
                    print(f"      [429 Rate Limit] {model_name} bận, đợi {wait_sec}s rồi thử lại (lần {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                    continue
                else:
                    err_msg = resp.text[:120]
                    print(f"      [{resp.status_code}] {model_name} báo lỗi: {err_msg}")
                    break  # Đổi sang model tiếp theo
            except Exception as e:
                print(f"      [Exception] {model_name}: {e}")
                time.sleep(2)

    raise RuntimeError("Tất cả các model Free trên OpenRouter hiện đang bận hoặc quá tải!")

def process_single_page(pdf_path: str, page_idx: int, total_pages: int, pages_dir: Path, api_key: str, dpi: int, models_list: list = None) -> dict:
    """Xử lý trích xuất cho 1 trang duy nhất, có cache."""
    page_num = page_idx + 1
    page_file_md = pages_dir / f"page_{page_num:03d}.md"
    page_file_json = pages_dir / f"page_{page_num:03d}.json"

    # Kiểm tra Cache: Nếu file đã tồn tại và không rỗng -> Bỏ qua
    if page_file_md.exists() and page_file_md.stat().st_size > 50:
        print(f"⏭️  [Trang {page_num:02d}/{total_pages}] Đã có trong cache, bỏ qua.")
        with open(page_file_md, "r", encoding="utf-8") as f:
            content = f.read()
        return {"page": page_num, "status": "cached", "chars": len(content)}

    t0 = time.time()
    print(f"🚀 [Trang {page_num:02d}/{total_pages}] Bắt đầu trích xuất...")
    
    # Mở doc độc lập trong thread để thread-safe
    with fitz.open(pdf_path) as doc:
        img_bytes = render_page_to_png(doc, page_idx, dpi=dpi)

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    
    try:
        content = call_openrouter_with_retry(img_b64, api_key, models_list=models_list)
        if not content or len(content.strip()) < 20:
            raise ValueError(f"Mô hình trả về nội dung rỗng hoặc không hợp lệ ({repr(content)[:50]})")
            
        dur = time.time() - t0

        # Ghi file MD
        with open(page_file_md, "w", encoding="utf-8") as f:
            f.write(content)

        # Ghi file JSON metadata
        meta = {
            "page_number": page_num,
            "source_file": Path(pdf_path).name,
            "chars": len(content),
            "duration_sec": round(dur, 2),
            "timestamp": time.time()
        }
        with open(page_file_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ [Trang {page_num:02d}/{total_pages}] Hoàn thành trong {dur:.1f}s ({len(content)} ký tự)")
        return {"page": page_num, "status": "success", "chars": len(content), "dur": dur}

    except Exception as e:
        dur = time.time() - t0
        # Dọn dẹp file rác 0-byte nếu có
        if page_file_md.exists() and page_file_md.stat().st_size == 0:
            try:
                page_file_md.unlink()
            except Exception:
                pass
        print(f"❌ [Trang {page_num:02d}/{total_pages}] Thất bại ({dur:.1f}s): {e}")
        return {"page": page_num, "status": "error", "error": str(e), "dur": dur}

def assemble_full_document(out_dir: Path, total_pages: int, pdf_name: str):
    """Ghép nối tất cả các trang markdown thành 1 văn bản BCTC hoàn chỉnh có neo trang."""
    pages_dir = out_dir / "pages"
    full_md_path = out_dir / f"{Path(pdf_name).stem}_full_extracted.md"
    
    print(f"\n📑 Đang tổng hợp {total_pages} trang vào: {full_md_path}...")
    full_content = []
    full_content.append(f"# BÁO CÁO TÀI CHÍNH TOÀN VĂN: {pdf_name}\n")
    full_content.append(f"> Tài liệu nguồn: `{pdf_name}` | Tổng số trang: {total_pages}\n\n---\n")

    successful_pages = 0
    for p in range(1, total_pages + 1):
        page_md = pages_dir / f"page_{p:03d}.md"
        if page_md.exists():
            with open(page_md, "r", encoding="utf-8") as f:
                text = f.read().strip()
            full_content.append(f"\n\n<!-- PAGE_START: {p} -->\n## [Trang {p}/{total_pages}]\n\n")
            full_content.append(text)
            full_content.append(f"\n\n<!-- PAGE_END: {p} -->\n\n---\n")
            successful_pages += 1
        else:
            full_content.append(f"\n\n## [Trang {p}/{total_pages} - CHƯA TRÍCH XUẤT]\n\n---\n")

    with open(full_md_path, "w", encoding="utf-8") as f:
        f.write("".join(full_content))

    print(f"🎉 Đã tổng hợp thành công {successful_pages}/{total_pages} trang!")
    print(f"💾 File lưu tại: {full_md_path} ({full_md_path.stat().st_size / 1024:.1f} KB)")

def run_pipeline(pdf_path: str, out_dir: str, max_workers: int = 2, start_page: int = 1, end_page: int = None, dpi: int = 140, model: str = None):
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("❌ Lỗi: Chưa tìm thấy OPEN_ROUTER_API_KEY trong file .env!")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"❌ Lỗi: Không tìm thấy file PDF: {pdf_path}")
        sys.exit(1)

    out_path = Path(out_dir)
    pages_dir = out_path / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_doc_pages = len(doc)

    start_idx = max(0, start_page - 1)
    end_idx = min(total_doc_pages, end_page) if end_page else total_doc_pages
    pages_to_process = list(range(start_idx, end_idx))
    count = len(pages_to_process)

    active_models = [model] if model else FREE_MODELS

    print("==================================================================")
    print("🚀 PIPELINE TRÍCH XUẤT SCANNED BCTC QUA OPENROUTER FREE VLM")
    print("==================================================================")
    print(f"📄 Tài liệu: {pdf_path}")
    print(f"📊 Tổng số trang tài liệu: {total_doc_pages}")
    print(f"🎯 Phạm vi xử lý: Trang {start_idx + 1} -> Trang {end_idx} ({count} trang)")
    print(f"⚡ Số luồng xử lý song song (Workers): {max_workers}")
    print(f"🤖 Model sử dụng: {active_models[0]} (kèm fallback dự phòng)")
    print(f"🖼️ Độ phân giải (DPI): {dpi}")
    print(f"📁 Thư mục lưu kết quả: {out_path}")
    print("==================================================================\n")

    t_start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_page, pdf_path, idx, total_doc_pages, pages_dir, api_key, dpi, active_models): idx
            for idx in pages_to_process
        }
        for f in as_completed(futures):
            res = f.result()
            results.append(res)

    total_time = time.time() - t_start
    print("\n==================================================================")
    print("🏁 KẾT THÚC QUÁ TRÌNH TRÍCH XUẤT")
    print(f"⏱️ Tổng thời gian chạy: {total_time:.2f} giây (~{total_time/60:.2f} phút)")
    success_count = sum(1 for r in results if r.get("status") in ["success", "cached"])
    print(f"✅ Trang thành công: {success_count}/{count}")
    print("==================================================================")

    # Ghép nối file toàn văn
    assemble_full_document(out_path, total_doc_pages, Path(pdf_path).name)

def main():
    parser = argparse.ArgumentParser(description="Pipeline trích xuất Scanned BCTC đa luồng qua OpenRouter Free VLM")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help=f"Đường dẫn file PDF (mặc định: {DEFAULT_PDF})")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"Thư mục lưu kết quả (mặc định: {DEFAULT_OUT_DIR})")
    parser.add_argument("--workers", type=int, default=2, help="Số luồng song song (khuyến nghị 2 hoặc 3 cho free tier)")
    parser.add_argument("--start", type=int, default=1, help="Trang bắt đầu (1-based)")
    parser.add_argument("--end", type=int, default=None, help="Trang kết thúc (1-based, None = hết tài liệu)")
    parser.add_argument("--dpi", type=int, default=140, help="Độ phân giải DPI khi render trang")
    parser.add_argument("--model", default=None, help="Chỉ định model cụ thể (ví dụ: google/gemma-4-31b-it:free)")
    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.pdf,
        out_dir=args.out,
        max_workers=args.workers,
        start_page=args.start,
        end_page=args.end,
        dpi=args.dpi,
        model=args.model
    )

if __name__ == "__main__":
    main()
