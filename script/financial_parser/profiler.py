"""Page-level PDF profiling and language evidence.

This module deliberately does not parse tables.  It only decides whether the
embedded PDF text is safe enough to send to the vector-text Docling path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
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


def _likely_tabular(words: List[Any]) -> bool:
    """Detect repeated rows with two or more numeric cells using native bboxes.

    This is intentionally a weak signal. It is used only to catch a Docling
    result that turned an evidently tabular native page into loose paragraphs.
    """
    rows: List[Tuple[float, int]] = []
    for word in sorted(words, key=lambda item: (float(item[1]), float(item[0]))):
        try:
            y_mid = (float(word[1]) + float(word[3])) / 2
            text = str(word[4]).strip()
        except (IndexError, TypeError, ValueError):
            continue
        is_number = bool(NUMBER_PATTERN.fullmatch(text))
        if not rows or abs(rows[-1][0] - y_mid) > 4.0:
            rows.append((y_mid, int(is_number)))
        elif is_number:
            rows[-1] = (rows[-1][0], rows[-1][1] + 1)
    return sum(number_count >= 2 for _, number_count in rows) >= 2


def profile_page(page: Any, pdf_page: int, config: ParserConfig) -> PageProfile:
    """Profile physical page content and assign a conservative page class."""
    raw_text = page.get_text("text") or ""
    words = page.get_text("words") or []
    cleaned = raw_text.strip()
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
        likely_tabular=_likely_tabular(words),
        reasons=reasons,
    )
