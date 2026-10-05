"""
grouper.py - Method 5 LLM Semantic Grouping, Safe Batching, and Heuristic Fallback
"""

from __future__ import annotations

import json
import re
import time
from typing import List

import requests

from financial_chunker.method5.common import (
    BATCH_UNIT_SIZE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    StructuralUnit,
    TARGET_GROUP_MAX_TOKENS,
)
from financial_chunker.method5.headings import is_top_level_heading


def request_llm_semantic_grouping(
    batch_units: List[StructuralUnit],
    ticker: str,
    fiscal_year: int,
) -> List[List[str]]:
    """
    Gửi batch 15-20 Structural Units cho DeepSeek-Chat để xác định ranh giới gom cụm ngữ nghĩa.
    LLM chỉ trả về danh sách ID, không sinh hay sửa nội dung.
    """
    if not DEEPSEEK_API_KEY:
        return fallback_heuristic_grouping(batch_units)

    items_payload = [
        {
            "id": u.unit_id,
            "type": u.unit_type,
            "tokens": u.tokens,
            "preview": u.preview,
        }
        for u in batch_units
    ]

    system_prompt = (
        "You are an expert financial document structural chunking assistant. "
        "You are given an ordered sequence of structural units from a 10-K report: "
        "H (Heading), P (Paragraph), T (Table), FN (Footnote). "
        "Your task: Group consecutive units into semantically cohesive chunks.\n\n"
        "Rules:\n"
        "1. SIZE CONSTRAINT: Target 400-900 tokens per group. Hard ceiling: 1100 tokens total.\n"

        "2. TABLE BINDING: A Table (T) binds with:\n"
        "   (a) AT MOST the 1-2 immediately preceding Paragraphs (P) that introduce it "
        "(typically ending with ':' or containing 'as follows', 'presented below', 'the following table').\n"
        "   (b) ALL immediately following Footnotes (FN) that belong to it.\n"
        "   Group them together ONLY if total combined tokens <= 1100.\n"

        "3. LARGE TABLE: A Table with tokens > 900 MUST NOT absorb preceding intro paragraphs, "
        "but MAY keep its directly attached Footnotes if total <= 1100.\n"

        "4. POST-TABLE ISOLATION: After a Table (T) and its Footnotes (FN) end, any subsequent "
        "independent Paragraph (P) or Table (T) MUST start a new group.\n"

        "5. NARRATIVE GROUPING: For consecutive content Paragraphs (P) under the same section, "
        "accumulate them into a single group until reaching 400-900 tokens. Avoid single-paragraph groups.\n"

        "6. HEADING BINDING: Consecutive Headings (H) MUST be prepended to the SAME group "
        "as the subsequent content (P or T) they introduce. Multiple headings per group are expected.\n"

        "7. NO ORPHAN HEADINGS: Within the body of the batch, a group consisting ONLY of Headings [H] "
        "is forbidden. However, if one or more Headings appear at the very end of the batch with no "
        "subsequent P or T, output them as their own group [H] (our pipeline will merge them with the next batch).\n"

        "8. TOP-LEVEL BOUNDARY: Never group across major boundaries. A major boundary is a Heading "
        "starting with NOTE, ITEM, PART, 'FORM 10-K', or containing 'CONSOLIDATED STATEMENTS' / "
        "'CONSOLIDATED BALANCE SHEETS'. Sub-headings (years, action categories) are NOT major boundaries.\n"

        "9. FOOTNOTE BINDING: Each Footnote (FN) belongs strictly to the Table (T) immediately preceding it. "
        "Once an Intro Paragraph (P ending with ':' or containing 'as follows'/'presented below') appears, "
        "the previous Table cluster is permanently closed. Footnotes must never cross an Intro Paragraph.\n"

        "10. STRICT ADJACENCY: Group only adjacent units in exact input order. Never skip, reorder, "
        "or drop units.\n"

        "11. OUTPUT FORMAT: Respond ONLY with a valid JSON object with a single key 'groups'. "
        "Do not include markdown code block wrappers (no ```json).\n"
        '{"groups": [["u0001", "u0002", "u0003"], ["u0004", "u0005"]]}\n'

        "12. VALIDATION: Every input unit ID must appear in exactly one group. Do not invent IDs."
    )

    user_prompt = (
        f"Company: {ticker} FY{fiscal_year} 10-K\n"
        f"Structural units to group:\n{json.dumps(items_payload, ensure_ascii=False, indent=2)}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    endpoint = f"{DEEPSEEK_BASE_URL}/chat/completions"

    for attempt in range(1, 4):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                clean_json = re.sub(r"^```(?:json)?\s*", "", content)
                clean_json = re.sub(r"\s*```$", "", clean_json).strip()
                parsed = json.loads(clean_json)
                groups = parsed.get("groups", [])
                if isinstance(groups, list) and groups:
                    all_returned_ids = [item for g in groups for item in g]
                    input_ids = [u.unit_id for u in batch_units]
                    if set(all_returned_ids) == set(input_ids):
                        return groups
            else:
                print(f"[LLM Warning] Attempt {attempt} returned HTTP {resp.status_code}: {resp.text[:120]}", flush=True)
        except Exception as e:
            print(f"[LLM Warning] Attempt {attempt} failed: {e}", flush=True)
        time.sleep(2 ** attempt)

    print(f"[{ticker}] LLM grouping failed for batch — falling back to heuristic.", flush=True)
    return fallback_heuristic_grouping(batch_units)


def _in_table_cluster(cur: List[StructuralUnit]) -> bool:
    """True nếu unit cuối là Table hoặc Footnote gần nhất — chưa nên cắt."""
    for u in reversed(cur):
        if u.unit_type == "T":
            return True
        if u.unit_type in ("P", "H"):
            return False
    return False


def split_into_safe_batches(
    units: List[StructuralUnit],
    max_size: int = BATCH_UNIT_SIZE,
) -> List[List[StructuralUnit]]:
    """
    Chia danh sách Structural Units thành các batch an toàn:
    - Bắt đầu batch mới tại top-level heading nếu batch hiện tại đã đạt ít nhất một nửa kích thước.
    - Giữ các đơn vị liên quan (Table + Intro + Footnotes) không bị chia cắt mù quáng.
    """
    batches: List[List[StructuralUnit]] = []
    cur: List[StructuralUnit] = []
    for u in units:
        if len(cur) >= max_size and (not _in_table_cluster(cur) or u.unit_type != "FN" or len(cur) >= max_size + 4):
            batches.append(cur)
            cur = []
        elif (
            u.unit_type == "H"
            and is_top_level_heading(u.text)
            and len(cur) >= max_size // 2
            and not _in_table_cluster(cur)
        ):
            batches.append(cur)
            cur = []
        cur.append(u)
    if cur:
        batches.append(cur)
    return batches


def fallback_heuristic_grouping(batch_units: List[StructuralUnit]) -> List[List[str]]:
    """
    Financial Semantic Heuristic Grouping with Section Boundary Guard:
    1. Heading Accumulation: Không flush khi đang gom các heading liên tiếp (H1, H2, H3).
       Chỉ flush khi nhóm đã có nội dung (P, T, FN) để bắt đầu một section mới.
    2. Table Affinity: Bảng giữ chặt đoạn giới thiệu (P) và footnotes (FN).
    3. Post-table isolation: Bảng kế tiếp hoặc đoạn văn độc lập tách riêng.
    4. Target chunk size: 400 - 900 tokens. Max 1100 tokens.
    """
    groups: List[List[str]] = []
    curr_group: List[StructuralUnit] = []
    curr_tokens = 0

    def flush_group():
        nonlocal curr_group, curr_tokens
        if curr_group:
            groups.append([u.unit_id for u in curr_group])
            curr_group = []
            curr_tokens = 0

    for u in batch_units:
        # Rule 1: Heading boundary guard - Chỉ flush nếu nhóm hiện tại đã có nội dung thân (P, T, FN)
        if u.unit_type == "H":
            if any(item.unit_type in ("P", "T", "FN") for item in curr_group):
                flush_group()
            curr_group.append(u)
            curr_tokens += u.tokens
            continue

        # Rule 2: Large standalone table (> 800 tokens)
        if u.unit_type == "T" and u.tokens > 800:
            flush_group()
            groups.append([u.unit_id])
            continue

        # Rule 3: Table affinity and post-table isolation
        has_table_in_group = any(item.unit_type == "T" for item in curr_group)
        if has_table_in_group:
            if u.unit_type == "T":
                # Multiple tables should form separate atomic chunks
                flush_group()
                curr_group.append(u)
                curr_tokens = u.tokens
                continue
            elif u.unit_type == "P":
                p_text_clean = u.text.strip()
                # Nếu là đoạn văn ngắn kết luận (< 120 tokens) không có dấu ':', giữ lại với bảng làm post-table note
                if (
                    u.tokens < 120 and
                    not p_text_clean.endswith(":") and
                    not re.search(r"(?:follow|follows|as follows)\s*:\s*$", p_text_clean, re.I) and
                    curr_tokens + u.tokens <= TARGET_GROUP_MAX_TOKENS
                ):
                    curr_group.append(u)
                    curr_tokens += u.tokens
                    continue
                else:
                    flush_group()
                    curr_group.append(u)
                    curr_tokens = u.tokens
                    continue
            elif u.unit_type == "FN":
                if curr_tokens + u.tokens <= TARGET_GROUP_MAX_TOKENS:
                    curr_group.append(u)
                    curr_tokens += u.tokens
                else:
                    flush_group()
                    curr_group.append(u)
                    curr_tokens = u.tokens
                continue

        # Rule 4: Normal token limit
        if curr_tokens + u.tokens > TARGET_GROUP_MAX_TOKENS and curr_group:
            flush_group()
            curr_group = [u]
            curr_tokens = u.tokens
        else:
            curr_group.append(u)
            curr_tokens += u.tokens

    flush_group()
    return groups
