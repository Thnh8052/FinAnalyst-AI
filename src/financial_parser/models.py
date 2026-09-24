"""Typed data contracts exchanged by profiling, parsing, QC and chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PageClass(str, Enum):
    NATIVE_TEXT = "native_text"
    IMAGE_ONLY = "image_only"
    OCR_LAYER = "ocr_layer"
    HYBRID = "hybrid"
    UNCERTAIN = "uncertain"


class EngineName(str, Enum):
    DOCLING = "docling"
    DEEPSEEK_VLM = "deepseek_vlm"
    GEMINI_VLM = "gemini_vlm"
    LOCAL_VLM = "local_vlm"
    LLAMAPARSE_VLM = "llamaparse_vlm"
    TATR = "tatr"
    MANUAL_REVIEW = "manual_review"


class QCStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class LanguageEvidence:
    primary: str = "unknown"  # vi | en | mixed | unknown
    confidence: float = 0.0
    detected_from: str = "native_text"
    vi_score: int = 0
    en_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PageProfile:
    pdf_page: int
    page_class: PageClass
    language: LanguageEvidence
    raw_text: str
    char_count: int
    word_count: int
    text_density: float
    image_count: int
    max_image_coverage: float
    total_image_coverage: float
    replacement_char_ratio: float
    private_use_ratio: float
    control_char_ratio: float
    type3_font_count: int
    has_suspicious_font: bool
    likely_tabular: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self, include_raw_text: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        result["page_class"] = self.page_class.value
        result["language"] = self.language.to_dict()
        if not include_raw_text:
            result.pop("raw_text", None)
        return result


@dataclass
class RouteDecision:
    pdf_page: int
    engine: EngineName
    reason: List[str]
    attempt: int = 1
    fallback_from: Optional[EngineName] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["engine"] = self.engine.value
        result["fallback_from"] = self.fallback_from.value if self.fallback_from else None
        return result


@dataclass
class QCResult:
    status: QCStatus
    source_text_recall: Optional[float] = None
    source_text_precision: Optional[float] = None
    numeric_recall: Optional[float] = None
    numeric_precision: Optional[float] = None
    table_shape_pass: bool = True
    table_count: int = 0
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result
