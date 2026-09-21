"""
Script Trích xuất Báo cáo Tài chính dạng Scanned PDF (Thế Giới Di Động - MWG 2025)
Chiến lược Hybrid: Thử Local (Qwen2.5-VL qua Ollama) -> Tự động Fallback sang OpenRouter (Model Free)
"""

import os
import sys
import time
import base64
import json
import argparse
import requests
from dotenv import load_dotenv

# Đảm bảo in tiếng Việt trên console Windows không bị UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "qwen2.5vl:3b"
IMAGE_PATH = "data/ocr_test_samples/test_mwg_balance_sheet.png"
OUTPUT_MD = "data/ocr_test_samples/mwg_balance_sheet_extracted.md"

# Danh sách model Vision MIỄN PHÍ trên OpenRouter theo thứ tự ưu tiên
OPENROUTER_FREE_MODELS = [
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

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def check_ollama_ready() -> bool:
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=2)
        if res.status_code == 200:
            models = [m.get("name", "") for m in res.json().get("models", [])]
            return any(LOCAL_MODEL in m for m in models)
    except Exception:
        pass
    return False

def run_local_ollama(img_b64: str) -> str:
    print(f"\n💻 [LOCAL] Đang chạy với Ollama ({LOCAL_MODEL})...")
    payload = {
        "model": LOCAL_MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 2048,
            "num_gpu": 99
        }
    }
    collected = []
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama trả về HTTP {resp.status_code}: {resp.text}")
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                tok = chunk.get("response", "")
                sys.stdout.write(tok)
                sys.stdout.flush()
                collected.append(tok)
    return "".join(collected)

def run_openrouter_free(img_b64: str) -> str:
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("Chưa cấu hình OPEN_ROUTER_API_KEY trong file .env!")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FinAnalyst-AI",
        "X-Title": "FinAnalyst-AI"
    }

    for model in OPENROUTER_FREE_MODELS:
        print(f"\n🌐 [OPENROUTER FREE] Đang gọi model: {model}...")
        payload = {
            "model": model,
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
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"✅ Thành công với {model}!")
                print(content)
                return content
            elif resp.status_code == 429:
                print(f"⚠️ {model} bị rate limit (429), thử model free tiếp theo...")
                time.sleep(1)
                continue
            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text[:120])
                print(f"⚠️ {model} lỗi ({resp.status_code}): {err_msg}")
        except Exception as e:
            print(f"⚠️ {model} ngoại lệ: {e}")

    raise RuntimeError("Tất cả các model Free trên OpenRouter hiện đang bận hoặc quá tải!")

def main():
    parser = argparse.ArgumentParser(description="Trích xuất BCTC Scanned PDF bằng VLM (Local-First + Free Fallback)")
    parser.add_argument("--provider", choices=["auto", "local", "openrouter"], default="auto",
                        help="auto: Thử Local trước, lỗi thì fallback OpenRouter Free. local: Chỉ chạy local. openrouter: Chỉ gọi OpenRouter Free.")
    args = parser.parse_args()

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Không tìm thấy file ảnh: {IMAGE_PATH}")
        sys.exit(1)

    print("================================================================")
    print("🚀 BẮT ĐẦU TRÍCH XUẤT BCTC SCANNED PDF")
    print(f"📂 Nguồn ảnh: {IMAGE_PATH}")
    print(f"⚙️ Chế độ: {args.provider.upper()}")
    print("================================================================")

    img_b64 = encode_image(IMAGE_PATH)
    start_time = time.time()
    result_text = None

    if args.provider in ["auto", "local"]:
        if check_ollama_ready():
            try:
                result_text = run_local_ollama(img_b64)
            except Exception as e:
                print(f"\n❌ Local Ollama gặp lỗi: {e}")
                if args.provider == "local":
                    sys.exit(1)
        else:
            print("ℹ️ Local Ollama chưa sẵn sàng (chưa mở hoặc chưa tải model).")
            if args.provider == "local":
                print("👉 Vui lòng chạy lệnh: ollama run qwen2.5vl:3b")
                sys.exit(1)

    if not result_text and args.provider in ["auto", "openrouter"]:
        print("\n🔀 Kích hoạt Fallback sang OpenRouter API (Model Miễn phí)...")
        try:
            result_text = run_openrouter_free(img_b64)
        except Exception as e:
            print(f"\n❌ Fallback OpenRouter thất bại: {e}")
            sys.exit(1)

    if result_text:
        elapsed = time.time() - start_time
        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write(result_text)
        print("\n================================================================")
        print(f"⏱️ Tổng thời gian: {elapsed:.2f} giây")
        print(f"💾 Kết quả đã lưu tại: {OUTPUT_MD}")
        print("================================================================")

if __name__ == "__main__":
    main()
