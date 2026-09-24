"""
method4_boundary_tagging.py

Phương pháp 4: LLM Semantic Boundary Tagging (Split-Point Identification)
- Áp dụng cấu trúc Prompt Boundary Tagging:
    - Văn bản được chia thành các base chunks đánh dấu <|start_chunk_X|> và <|end_chunk_X|>.
    - LLM xác định điểm chuyển chủ đề: 'split_after: 3, 5'.
    - Yêu cầu bắt buộc: 'YOU MUST RESPOND WITH AT LEAST ONE SPLIT'.
    - Quy mô mục tiêu: min_words (250) đến max_words (550).
- Chạy qua ThreadPoolExecutor (8 workers) gọi DeepSeek-Chat.
- Đo lường chính xác tỷ lệ vỡ bảng và phân bố token khi LLM cắt trên raw text có gắn tag.
- Lưu trữ riêng biệt vào: output_chunking/<company>/method4_boundary_tagging/
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import requests
import tiktoken
from dotenv import load_dotenv

from financial_chunker.models import ChunkMethod, ChunkType, FinancialChunk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

from financial_chunker.config import COMPANY_SPECS, OUTPUT_CHUNKING_ROOT, OUTPUT_PARSING_ROOT

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = "deepseek-chat"

ENCODING_NAME = "cl100k_base"
BATCH_CHUNKS_SIZE = 12
MIN_WORDS = 250
MAX_WORDS = 550

_CHUNKING_PROMPT = """
You are an assistant specialized in splitting text into semantically consistent sections.

Following is the document text:
<document>
{document_text}
</document>

<instructions>
Instructions:
    1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
    2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
    3. Each output section should be roughly {min_words} to {max_words} words when merged.
    4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
    5. The chunks must be listed in ascending order.
    6. Provide your response in the form: 'split_after: 3, 5'.
</instructions>

Respond only with the IDs of the chunks where you believe a split should occur.
YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
""".strip()


@dataclass
class BaseTaggedUnit:
    global_id: int
    local_id: int
    page: int
    text: str
    word_count: int
    qc_status: str = "pass"


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def is_table_separator(line: str) -> bool:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return all(re.match(r"^:?-+:?$", c) for c in cells if c)


def detect_table_fragmentation(chunk_text: str) -> Tuple[bool, List[str]]:
    lines = chunk_text.splitlines()
    reasons: List[str] = []
    in_table = False
    table_has_header = False

    for i, line in enumerate(lines):
        if is_table_row(line):
            if not in_table:
                in_table = True
                if i + 1 < len(lines) and is_table_separator(lines[i + 1]):
                    table_has_header = True
                else:
                    table_has_header = False
                    reasons.append(f"starts_mid_table_at_line_{i}")
        else:
            if in_table:
                if not table_has_header:
                    reasons.append(f"table_lacks_header_before_line_{i}")
                in_table = False
                table_has_header = False

    return len(reasons) > 0, reasons


def extract_base_tagged_units(pages_dir: Path) -> List[BaseTaggedUnit]:
    """
    Trích xuất các base text units (khoảng 50-120 từ) từ các trang tài liệu.
    """
    json_files = sorted(
        pages_dir.glob("page_*.json"),
        key=lambda p: int(re.search(r"page_(\d+)\.json", p.name).group(1)),
    )

    base_units: List[BaseTaggedUnit] = []
    global_counter = 0

    for jf in json_files:
        pdf_page = int(re.search(r"page_(\d+)\.json", jf.name).group(1))
        md_file = pages_dir / f"page_{pdf_page:03d}.md"

        if md_file.exists():
            page_text = md_file.read_text(encoding="utf-8")
        else:
            with open(jf, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            blocks = p_data.get("page", {}).get("blocks", [])
            page_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))

        with open(jf, "r", encoding="utf-8") as f:
            p_json = json.load(f)
        page_qc = p_json.get("page", {}).get("qc", {}).get("status", "pass")

        # Chia nhỏ text theo đoạn văn (Paragraphs / Table Blocks)
        paragraphs = re.split(r"\n\s*\n", page_text.strip())
        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            words = p_clean.split()
            # Nếu đoạn quá dài (>180 words), cắt thành các đoạn con ~100 words
            if len(words) > 180:
                for w_idx in range(0, len(words), 110):
                    sub_words = words[w_idx : w_idx + 110]
                    sub_text = " ".join(sub_words)
                    global_counter += 1
                    base_units.append(
                        BaseTaggedUnit(
                            global_id=global_counter,
                            local_id=0,
                            page=pdf_page,
                            text=sub_text,
                            word_count=len(sub_words),
                            qc_status=page_qc,
                        )
                    )
            else:
                global_counter += 1
                base_units.append(
                    BaseTaggedUnit(
                        global_id=global_counter,
                        local_id=0,
                        page=pdf_page,
                        text=p_clean,
                        word_count=len(words),
                        qc_status=page_qc,
                    )
                )

    return base_units


def request_llm_split_after(
    batch_units: List[BaseTaggedUnit],
    ticker: str,
    fiscal_year: int,
) -> List[int]:
    """
    Gửi văn bản đã gắn tag <|start_chunk_X|> cho DeepSeek-Chat và phân tích phản hồi 'split_after: 3, 5'.
    """
    n_units = len(batch_units)
    if n_units <= 1:
        return [1]

    # Gắn local IDs từ 1 đến N
    tagged_lines: List[str] = []
    for idx, u in enumerate(batch_units, 1):
        u.local_id = idx
        tagged_lines.append(f"<|start_chunk_{idx}|>\n{u.text}\n<|end_chunk_{idx}|>")

    doc_text = "\n\n".join(tagged_lines)
    prompt = _CHUNKING_PROMPT.format(
        document_text=doc_text,
        min_words=MIN_WORDS,
        max_words=MAX_WORDS,
    )

    if not DEEPSEEK_API_KEY:
        return fallback_split_after(batch_units)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    endpoint = f"{DEEPSEEK_BASE_URL}/chat/completions"

    for attempt in range(1, 3):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                raw_ans = data["choices"][0]["message"]["content"].strip()
                # Parse 'split_after: 3, 5'
                m = re.search(r"split_after:\s*([0-9,\s]+)", raw_ans, re.I)
                if m:
                    nums_str = m.group(1)
                    splits = [int(x.strip()) for x in re.findall(r"\d+", nums_str)]
                    valid_splits = sorted(list({s for s in splits if 1 <= s < n_units}))
                    if valid_splits:
                        return valid_splits
                else:
                    # Tìm các số trong câu trả lời nếu không có tiền tố
                    nums = [int(x) for x in re.findall(r"\b\d+\b", raw_ans)]
                    valid_splits = sorted(list({s for s in nums if 1 <= s < n_units}))
                    if valid_splits:
                        return valid_splits
        except Exception:
            pass
        time.sleep(1)

    return fallback_split_after(batch_units)


def fallback_split_after(batch_units: List[BaseTaggedUnit]) -> List[int]:
    """
    Fallback chia tách nếu API gặp sự cố: ngắt khi từ tích lũy vượt ngưỡng ~350 từ.
    """
    splits: List[int] = []
    curr_words = 0
    n = len(batch_units)

    for idx, u in enumerate(batch_units[:-1], 1):
        curr_words += u.word_count
        if curr_words >= 350:
            splits.append(idx)
            curr_words = 0

    if not splits and n > 1:
        # Prompt yêu cầu: YOU MUST RESPOND WITH AT LEAST ONE SPLIT
        splits.append(n // 2)

    return splits


def run_method4_for_company(company_spec: Dict[str, Any]) -> Dict[str, Any]:
    dir_name = company_spec["dir_name"]
    ticker = company_spec["ticker"]
    fiscal_year = company_spec["fiscal_year"]

    company_parsing_dir = OUTPUT_PARSING_ROOT / dir_name
    pages_dir = company_parsing_dir / "pages"

    if not pages_dir.exists():
        return {"company": dir_name, "status": "skipped_missing_pages", "chunk_count": 0}

    enc = tiktoken.get_encoding(ENCODING_NAME)
    base_units = extract_base_tagged_units(pages_dir)

    # Chia thành các batch 12 units
    batches = [
        base_units[i : i + BATCH_CHUNKS_SIZE]
        for i in range(0, len(base_units), BATCH_CHUNKS_SIZE)
    ]
    total_batches = len(batches)

    print(
        f"[{ticker}] Method 4 Boundary Tagging: {len(base_units)} base units, "
        f"Batch size: {BATCH_CHUNKS_SIZE} (Total batches: {total_batches})",
        flush=True,
    )

    def process_batch(item: Tuple[int, List[BaseTaggedUnit]]) -> Tuple[int, List[int]]:
        idx, b = item
        splits = request_llm_split_after(b, ticker, fiscal_year)
        return idx, splits

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        batch_results = list(executor.map(process_batch, enumerate(batches)))

    batch_results.sort(key=lambda x: x[0])

    # Ghép các base units theo các split points đã xác định
    final_chunks: List[FinancialChunk] = []
    chunk_counter = 0

    for b_idx, (_, splits) in enumerate(batch_results):
        batch = batches[b_idx]
        split_set = set(splits)

        curr_slice: List[BaseTaggedUnit] = []
        for local_id, u in enumerate(batch, 1):
            curr_slice.append(u)
            if local_id in split_set or local_id == len(batch):
                # Tạo 1 chunk mới
                chunk_counter += 1
                merged_text = "\n\n".join(x.text for x in curr_slice).strip()
                tokens = len(enc.encode(merged_text))
                source_pages = sorted(list({x.page for x in curr_slice}))
                qc_warning = any(x.qc_status == "warning" for x in curr_slice)

                is_frag, _ = detect_table_fragmentation(merged_text)

                chunk_id = f"{ticker.lower()}_{fiscal_year}_m4_btag_{chunk_counter:04d}"
                c = FinancialChunk(
                    chunk_id=chunk_id,
                    document_id=dir_name,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    chunk_method=ChunkMethod.METHOD4_BOUNDARY_TAGGING.value,
                    chunk_type=ChunkType.FIXED_SIZE.value,
                    content=merged_text,
                    content_retrieval=merged_text,
                    content_generation=merged_text,
                    token_count=tokens,
                    source_pages=source_pages,
                    has_table_fragmentation=is_frag,
                    metadata={
                        "split_mode": "llm_split_after_tagging",
                        "has_qc_warning": qc_warning,
                        "source_verification_badge": (
                            f"⚠️ Lưu ý kiểm tra nguồn: Văn bản hoặc bảng số liệu tại Trang {source_pages[0]} "
                            "có cấu trúc phức tạp. Vui lòng đối chiếu với tài liệu gốc nếu cần."
                            if qc_warning else None
                        ),
                        "pdf_inspector": {
                            "pdf_page": source_pages[0],
                            "highlight_target": "text_block",
                        },
                    },
                )
                final_chunks.append(c)
                curr_slice = []

    # Thống kê kết quả
    token_counts = [c.token_count for c in final_chunks]
    frag_count = sum(1 for c in final_chunks if c.has_table_fragmentation)
    token_counts_sorted = sorted(token_counts)
    min_tokens = token_counts_sorted[0] if token_counts else 0
    max_tokens = token_counts_sorted[-1] if token_counts else 0
    mean_tokens = round(sum(token_counts) / len(token_counts), 2) if token_counts else 0
    median_tokens = token_counts_sorted[len(token_counts_sorted) // 2] if token_counts else 0

    # Lưu kết quả vào folder riêng
    comp_chunk_dir = OUTPUT_CHUNKING_ROOT / dir_name / "method4_boundary_tagging"
    comp_chunk_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = comp_chunk_dir / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in final_chunks:
            f.write(c.to_json() + "\n")

    summary_info = {
        "company": dir_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "method": ChunkMethod.METHOD4_BOUNDARY_TAGGING.value,
        "total_base_units": len(base_units),
        "total_chunks_created": len(final_chunks),
        "fragmented_table_chunks": frag_count,
        "fragmentation_rate": round(frag_count / len(final_chunks), 4) if final_chunks else 0.0,
        "llm_api_batches": total_batches,
        "token_distribution": {
            "min": min_tokens,
            "max": max_tokens,
            "mean": mean_tokens,
            "median": median_tokens,
        },
        "output_file": str(jsonl_path),
    }

    summary_path = comp_chunk_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_info, f, ensure_ascii=False, indent=2)

    print(
        f"[{ticker}] DONE Method 4: {len(final_chunks)} chunks | "
        f"Frag: {frag_count} ({summary_info['fragmentation_rate']*100:.1f}%) | "
        f"Tokens mean: {mean_tokens}, median: {median_tokens}.",
        flush=True,
    )

    return summary_info


def main() -> None:
    start_time = time.time()
    print("=================================================================", flush=True)
    print("BẮT ĐẦU CHẠY PHƯƠNG PHÁP 4: LLM BOUNDARY TAGGING (SPLIT-AFTER)", flush=True)
    print("=================================================================", flush=True)

    results = []
    for spec in COMPANY_SPECS:
        print(f"\n>>> Đang xử lý: {spec['ticker']} ({spec['dir_name']}) ...", flush=True)
        res = run_method4_for_company(spec)
        results.append(res)

    total_time = round(time.time() - start_time, 2)
    print("\n=================================================================", flush=True)
    print(f"HOÀN TẤT PHƯƠNG PHÁP 4 CHO TẤT CẢ {len(COMPANY_SPECS)} HỒ SƠ DOANH NGHIỆP! Thời gian: {total_time}s", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
