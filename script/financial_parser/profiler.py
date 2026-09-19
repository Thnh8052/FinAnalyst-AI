"""Page-level PDF profiling and language evidence.

This module deliberately does not parse tables.  It only decides whether the
embedded PDF text is safe enough to send to the vector-text Docling path.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Tuple

from .config import ParserConfig
from .markdown_utils import NUMBER_PATTERN
from .models import LanguageEvidence, PageClass, PageProfile


VI_KEYWORDS = (
    "báo cáo tài chính", "bảng cân đối kế toán", "kết quả hoạt động kinh doanh",
    "lưu chuyển tiền tệ", "thuyết minh", "tài sản", "nợ phải trả",
    "vốn chủ sở hữu", "đơn vị tính", "công ty cổ phần", "kiểm toán",
)
EN_KEYWORDS = (
    "consolidated balance sheets", "statements of operations",
    "statements of income", "statements of cash flows",
    "notes to consolidated financial statements", "form 10-k", "form 10-q",
    "shareholders' equity", "stockholders' equity", "total assets",
)
VI_DIACRITICS = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def detect_language(text: str) -> LanguageEvidence:
    """Return language evidence for a page without forcing a binary label."""
    normalized = text.lower()
    vi_keyword_hits = sum(1 for item in VI_KEYWORDS if item in normalized)
    en_keyword_hits = sum(1 for item in EN_KEYWORDS if item in normalized)
    vi_diacritic_hits = len(VI_DIACRITICS.findall(normalized))
    vi_score = vi_keyword_hits * 5 + min(30, vi_diacritic_hits)
    en_score = en_keyword_hits * 5

    if vi_score == 0 and en_score == 0:
        return LanguageEvidence()
    if vi_score and en_score and abs(vi_score - en_score) <= 4:
        total = max(1, vi_score + en_score)
        return LanguageEvidence("mixed", round(max(vi_score, en_score) / total, 3), "native_text", vi_score, en_score)
    if vi_score > en_score:
        return LanguageEvidence("vi", round(vi_score / max(1, vi_score + en_score), 3), "native_text", vi_score, en_score)
    return LanguageEvidence("en", round(en_score / max(1, vi_score + en_score), 3), "native_text", vi_score, en_score)


def _coverage(rect: Any, page_area: float) -> float:
    try:
        return max(0.0, min(1.0, (rect.width * rect.height) / max(1.0, page_area)))
    except (AttributeError, TypeError):
        return 0.0


def _image_coverages(page: Any, page_area: float) -> List[float]:
    """Use image-info when available and fall back to xref rectangles."""
    coverages: List[float] = []
    try:
        for info in page.get_image_info(xrefs=True):
            bbox = info.get("bbox")
            if bbox is not None:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                coverages.append(max(0.0, min(1.0, width * height / max(1.0, page_area))))
        if coverages:
            return coverages
    except (AttributeError, RuntimeError, ValueError):
        pass

    try:
        for image in page.get_images(full=True):
            for rect in page.get_image_rects(image[0]):
                coverages.append(_coverage(rect, page_area))
    except (AttributeError, RuntimeError, ValueError):
        pass
    return coverages


def _font_signals(page: Any) -> Tuple[int, bool]:
    type3_count = 0
    suspicious = False
    try:
        for font in page.get_fonts(full=True):
            font_type = str(font[2]).lower() if len(font) > 2 else ""
            base_font = str(font[3]).lower() if len(font) > 3 else ""
            if "type3" in font_type or "type3" in base_font:
                type3_count += 1
            if "identity" in base_font or "cid" in base_font:
                suspicious = True
    except (AttributeError, RuntimeError, ValueError):
        pass
    return type3_count, suspicious


def _character_ratios(text: str) -> Tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    size = len(text)
    replacement = sum(char == "\ufffd" for char in text) / size
    private_use = sum("\ue000" <= char <= "\uf8ff" for char in text) / size
    control = sum(ord(char) < 32 and char not in "\n\r\t" for char in text) / size
    return replacement, private_use, control


def _likely_tabular(words: List[Any], page_height: float = 792.0) -> bool:
    """Detect repeated rows with vertical column alignment using native bboxes.

    A true financial data table has numbers aligned vertically in columns
    (similar x1 / right-aligned coordinates across distinct y rows).
    Requires either:
    1. At least 1 primary numeric column with >= 5 distinct rows, OR
    2. At least 2 numeric columns with >= 3 distinct rows each (sum of rows >= 6).
    """
    numeric_words: List[dict] = []
    for w in words:
        try:
            x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
            text = str(w[4]).strip()
        except (IndexError, TypeError, ValueError):
            continue
        # Filter top and bottom margins (header / footer page numbers)
        if y0 > page_height * 0.94 or y1 < page_height * 0.05:
            continue
        if NUMBER_PATTERN.fullmatch(text):
            numeric_words.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "x_mid": (x0 + x1) / 2,
                "y_mid": (y0 + y1) / 2,
            })

    if len(numeric_words) < 4:
        return False

    # Cluster numbers by x1 (right-aligned, standard in financial columns) with tolerance 8.0 pt
    clusters: defaultdict = defaultdict(list)
    sorted_by_x1 = sorted(numeric_words, key=lambda w: w["x1"])
    current_cluster: List[dict] = []
    for w in sorted_by_x1:
        if not current_cluster:
            current_cluster.append(w)
        else:
            avg_x1 = sum(item["x1"] for item in current_cluster) / len(current_cluster)
            if abs(w["x1"] - avg_x1) <= 8.0:
                current_cluster.append(w)
            else:
                if len(current_cluster) >= 3:
                    clusters[round(avg_x1, 1)].extend(current_cluster)
                current_cluster = [w]
    if len(current_cluster) >= 3:
        avg_x1 = sum(item["x1"] for item in current_cluster) / len(current_cluster)
        clusters[round(avg_x1, 1)].extend(current_cluster)

    # Count distinct rows (items separated by delta y >= 7.0 pt) in each column cluster
    valid_cols = []
    for col_x, items in clusters.items():
        items_by_y = sorted(items, key=lambda w: w["y_mid"])
        distinct_y_rows = []
        for it in items_by_y:
            if not distinct_y_rows or abs(distinct_y_rows[-1]["y_mid"] - it["y_mid"]) >= 7.0:
                distinct_y_rows.append(it)
        if len(distinct_y_rows) >= 3:
            valid_cols.append({"col_x": col_x, "rows": len(distinct_y_rows)})

    has_deep_single_col = any(c["rows"] >= 5 for c in valid_cols)
    has_multi_col = len(valid_cols) >= 2 and sum(c["rows"] for c in valid_cols) >= 6

    return has_deep_single_col or has_multi_col


RUNNING_MARGIN_PATTERN = re.compile(
    r"\|\s*\d+\s*$|Form\s*10-[KQ]|Annual\s*Report|Báo\s*cáo|Trang\s*\d+|Page\s*\d+|Exhibit\s*\d+(?:\.\d+)?|^\s*[-–—]?\s*\d+\s*[-–—]?\s*$",
    re.IGNORECASE,
)


def _extract_body_text_and_words(page: Any) -> Tuple[str, List[Any]]:
    """Extract page body text and words excluding top/bottom running headers and footers."""
    try:
        height = float(page.rect.height)
        blocks = page.get_text("blocks") or []
        body_blocks: List[str] = []
        num_blocks = len(blocks)
        for i, block in enumerate(blocks):
            text = str(block[4]).strip()
            y0, y1 = float(block[1]), float(block[3])
            is_margin_header = (
                (y1 < height * 0.08 and bool(RUNNING_MARGIN_PATTERN.search(text)))
                or (y1 < height * 0.04 and len(text.splitlines()) == 1)
            )
            is_trailing_block = i >= max(0, num_blocks - 2)
            is_isolated_page_num = bool(re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", text))
            is_copyright_footer = bool(re.search(r"all rights reserved|©\s*\d{4}|\(c\)\s*\d{4}", text, re.I))
            is_margin_footer = (
                (y0 > height * 0.94 and (bool(RUNNING_MARGIN_PATTERN.search(text)) or len(text.splitlines()) == 1))
                or (is_trailing_block and is_isolated_page_num and y0 > height * 0.40)
                or (is_trailing_block and is_copyright_footer and y0 > height * 0.40)
            )
            if not is_margin_header and not is_margin_footer:
                body_blocks.append(str(block[4]))

        all_words = page.get_text("words") or []
        num_words = len(all_words)
        body_words = []
        for i, w in enumerate(all_words):
            w_text = str(w[4]).strip()
            w_y0, w_y1 = float(w[1]), float(w[3])
            is_hdr_w = (
                (w_y1 < height * 0.08 and bool(RUNNING_MARGIN_PATTERN.search(w_text)))
                or (w_y1 < height * 0.04 and bool(RUNNING_MARGIN_PATTERN.search(w_text)))
            )
            is_trailing_w = i >= max(0, num_words - 4)
            is_page_num_w = bool(re.fullmatch(r"\d{1,4}", w_text)) and w_y0 > height * 0.40
            is_ftr_w = (
                (w_y0 > height * 0.94 and (bool(RUNNING_MARGIN_PATTERN.search(w_text)) or bool(re.fullmatch(r"\d+", w_text))))
                or (is_trailing_w and is_page_num_w)
            )
            if not is_hdr_w and not is_ftr_w:
                body_words.append(w)

        body_text = "".join(body_blocks).strip()
        if not body_text:
            body_text = (page.get_text("text") or "").strip()
            body_words = all_words
        return body_text, body_words
    except Exception:
        raw = (page.get_text("text") or "").strip()
        return raw, (page.get_text("words") or [])


def profile_page(page: Any, pdf_page: int, config: ParserConfig) -> PageProfile:
    """Profile physical page content and assign a conservative page class."""
    cleaned, words = _extract_body_text_and_words(page)
    page_area = max(1.0, page.rect.width * page.rect.height)
    coverages = _image_coverages(page, page_area)
    max_coverage = max(coverages, default=0.0)
    total_coverage = min(1.0, sum(coverages))
    replacement, private_use, control = _character_ratios(cleaned)
    type3_count, suspicious_font = _font_signals(page)
    char_count = len(cleaned)
    word_count = len(words)
    text_density = char_count / page_area
    reasons: List[str] = []

    invalid_text = replacement > 0.002 or private_use > 0.002 or control > 0.002
    # CID/Identity names are common in perfectly valid embedded Vietnamese and
    # CJK fonts. They are evidence to log, not enough on their own to reject a
    # native text page. Type 3 fonts are a stronger fidelity risk.
    hard_font_risk = type3_count > 0
    image_dominant = max_coverage >= config.ocr_layer_image_coverage
    enough_text = char_count >= config.min_native_chars and word_count >= 8

    if not enough_text and max_coverage < 0.20:
        page_class = PageClass.UNCERTAIN
        reasons.append("too_little_text_without_a_dominant_image")
    elif not enough_text:
        page_class = PageClass.IMAGE_ONLY
        reasons.append("no_reliable_native_text")
    elif image_dominant and (invalid_text or hard_font_risk or suspicious_font):
        page_class = PageClass.OCR_LAYER
        reasons.append("full_page_image_with_unreliable_embedded_text")
    elif image_dominant:
        page_class = PageClass.HYBRID
        reasons.append("full_page_image_with_selectable_text")
    elif invalid_text or hard_font_risk:
        page_class = PageClass.UNCERTAIN
        reasons.append("font_or_unicode_fidelity_risk")
    elif max_coverage > config.max_image_coverage_for_native:
        page_class = PageClass.HYBRID
        reasons.append("material_embedded_image_area")
    else:
        page_class = PageClass.NATIVE_TEXT
        reasons.append("native_text_and_geometry_are_usable")

    if type3_count:
        reasons.append(f"type3_fonts={type3_count}")
    if suspicious_font:
        reasons.append("cid_or_identity_font_seen")
    if invalid_text:
        reasons.append("invalid_unicode_or_control_characters_seen")

    return PageProfile(
        pdf_page=pdf_page,
        page_class=page_class,
        language=detect_language(cleaned),
        raw_text=cleaned,
        char_count=char_count,
        word_count=word_count,
        text_density=round(text_density, 7),
        image_count=len(coverages),
        max_image_coverage=round(max_coverage, 4),
        total_image_coverage=round(total_coverage, 4),
        replacement_char_ratio=round(replacement, 5),
        private_use_ratio=round(private_use, 5),
        control_char_ratio=round(control, 5),
        type3_font_count=type3_count,
        has_suspicious_font=suspicious_font,
        likely_tabular=_likely_tabular(words, page_height=float(page.rect.height) if hasattr(page, "rect") else 792.0),
        reasons=reasons,
    )
