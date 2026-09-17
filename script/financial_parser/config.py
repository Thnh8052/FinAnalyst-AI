"""Runtime configuration.  Secrets are only read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ParserConfig:
    """Configuration shared by the profiler, parser engines and quality gates."""

    vlm_provider: str = "deepseek"  # "deepseek" | "gemini"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    vlm_max_tokens: int = 8192
    vlm_concurrency: int = 4
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    render_dpi: int = 230
    docling_timeout_seconds: int = 180
    min_native_chars: int = 80
    max_image_coverage_for_native: float = 0.35
    ocr_layer_image_coverage: float = 0.80
    min_docling_numeric_recall: float = 0.90
    min_docling_numeric_precision: float = 0.970
    min_docling_text_recall: float = 0.90
    output_root: Path = Path("output_financial_parser")

    @classmethod
    def from_environment(
        cls,
        *,
        output_root: Optional[Path] = None,
        vlm_provider: Optional[str] = None,
        vlm_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        deepseek_model: Optional[str] = None,
        render_dpi: Optional[int] = None,
        vlm_concurrency: Optional[int] = None,
    ) -> "ParserConfig":
        """Load optional values without exposing API keys in logs or manifests."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        active_provider = (vlm_provider or os.getenv("VLM_PROVIDER", "deepseek")).strip().lower()
        if active_provider not in ("deepseek", "gemini"):
            active_provider = "deepseek"

        ds_model = (
            (vlm_model if active_provider == "deepseek" else None)
            or deepseek_model
            or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        ).strip()
        gem_model = (
            (vlm_model if active_provider == "gemini" else None)
            or gemini_model
            or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        ).strip()

        concurrency = vlm_concurrency or int(os.getenv("VLM_CONCURRENCY", "4"))

        return cls(
            vlm_provider=active_provider,
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/"),
            deepseek_model=ds_model,
            vlm_max_tokens=int(os.getenv("VLM_MAX_TOKENS", "8192")),
            vlm_concurrency=max(1, concurrency),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=gem_model,
            render_dpi=render_dpi or int(os.getenv("FINANCIAL_PARSER_DPI", "230")),
            output_root=output_root or Path(os.getenv("FINANCIAL_PARSER_OUT", "output_financial_parser")),
        )


