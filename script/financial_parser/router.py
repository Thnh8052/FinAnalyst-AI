"""Orchestrator for VAS/IFRS/SEC parsing with auditable fallback decisions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .canonical import markdown_to_canonical_page
from .config import ParserConfig
from .engines import DoclingEngine, EngineExecutionError, EngineUnavailableError, create_vlm_engine

from .markdown_utils import sanitize_markdown
from .models import EngineName, PageClass, PageProfile, QCResult, QCStatus, RouteDecision
from .profiler import profile_page
from .qc import evaluate_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contiguous_runs(page_numbers: Sequence[int]) -> List[Tuple[int, int]]:
    if not page_numbers:
        return []
    ordered = sorted(page_numbers)
    runs: List[Tuple[int, int]] = []
    first = previous = ordered[0]
    for current in ordered[1:]:
        if current == previous + 1:
            previous = current
        else:
            runs.append((first, previous))
            first = previous = current
    runs.append((first, previous))
    return runs


def _printed_page(page: Any, markdown: str) -> Optional[int]:
    signal = re.search(r"<!--\s*PRINTED_PAGE:\s*(\d{1,4})\s*-->", markdown, re.IGNORECASE)
    if signal:
        return int(signal.group(1))
    try:
        rect = page.rect
        footer_rect = type(rect)(0, rect.height * 0.88, rect.width, rect.height)
        footer = page.get_text("text", clip=footer_rect).strip()
        values = re.findall(r"(?<!\d)(\d{1,4})(?!\d)", footer)
        return int(values[-1]) if values else None
    except Exception:
        return None


class FinancialReportRouter:
    """Creates canonical page output while preserving route/QC provenance."""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config
        self.docling = DoclingEngine(config)
        self.vlm = create_vlm_engine(config)
        self.vlm_engine_name = (
            EngineName.GEMINI_VLM if config.vlm_provider == "gemini" else EngineName.DEEPSEEK_VLM
        )

    def _initial_route(self, profile: PageProfile) -> RouteDecision:
        if profile.page_class == PageClass.NATIVE_TEXT:
            return RouteDecision(profile.pdf_page, EngineName.DOCLING, list(profile.reasons))
        return RouteDecision(profile.pdf_page, self.vlm_engine_name, list(profile.reasons))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_page(
        self,
        *,
        pages_dir: Path,
        document_id: str,
        page: Any,
        profile: PageProfile,
        route: RouteDecision,
        markdown: str,
        qc: QCResult,
    ) -> None:
        md_path = pages_dir / f"page_{profile.pdf_page:03d}.md"
        json_path = pages_dir / f"page_{profile.pdf_page:03d}.json"
        md_path.write_text(markdown, encoding="utf-8")
        canonical = markdown_to_canonical_page(
            markdown=markdown,
            document_id=document_id,
            pdf_page=profile.pdf_page,
            printed_page=_printed_page(page, markdown),
            profile=profile,
            route=route,
            qc=qc,
        )
        self._write_json(json_path, canonical)

    @staticmethod
    def _review_record(profile: PageProfile, route: RouteDecision, qc: Optional[QCResult], error: str = "") -> Dict[str, Any]:
        return {
            "pdf_page": profile.pdf_page,
            "page_class": profile.page_class.value,
            "route": route.to_dict(),
            "qc": qc.to_dict() if qc else None,
            "error": error or None,
        }

    def _parse_with_vlm(
        self,
        *,
        page: Any,
        profile: PageProfile,
        route: RouteDecision,
    ) -> Tuple[str, QCResult, RouteDecision]:
        output = sanitize_markdown(self.vlm.parse_page(page, profile.pdf_page))
        final_route = route
        qc = evaluate_output(output, profile, self.vlm_engine_name, self.config)
        return output, qc, final_route

    def process(
        self,
        pdf_path: Path,
        *,
        start_page: int = 1,
        end_page: Optional[int] = None,
        output_dir: Optional[Path] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run profile → parse → QC → fallback and return an audit summary."""
        try:
            import fitz
        except ImportError as error:
            raise EngineUnavailableError("PyMuPDF is not installed. Install requirements-parsing.txt.") from error

        pdf_path = pdf_path.resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        root = (output_dir or self.config.output_root / pdf_path.stem).resolve()
        pages_dir = root / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        document_id = pdf_path.stem

        document = fitz.open(str(pdf_path))
        try:
            first = max(1, start_page)
            last = min(len(document), end_page or len(document))
            if first > last:
                raise ValueError(f"Invalid page range {start_page}..{end_page} for {len(document)} pages")

            profiles: Dict[int, PageProfile] = {
                page_number: profile_page(document[page_number - 1], page_number, self.config)
                for page_number in range(first, last + 1)
            }
            routes: Dict[int, RouteDecision] = {
                page_number: self._initial_route(profile)
                for page_number, profile in profiles.items()
            }
            active_vlm_model = (
                self.config.gemini_model if self.config.vlm_provider == "gemini" else self.config.deepseek_model
            )
            manifest: Dict[str, Any] = {
                "schema_version": "financial-parser-run-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "document": {
                    "document_id": document_id,
                    "source_file": str(pdf_path),
                    "file_sha256": _sha256(pdf_path),
                    "page_count": len(document),
                    "range": {"start": first, "end": last},
                },
                "configuration": {
                    "vlm_provider": self.config.vlm_provider,
                    "vlm_model": active_vlm_model,
                    "render_dpi": self.config.render_dpi,
                    "docling_timeout_seconds": self.config.docling_timeout_seconds,
                },
                "profiles": [profiles[number].to_dict() for number in sorted(profiles)],
                "initial_routes": [routes[number].to_dict() for number in sorted(routes)],
                "runtime": {
                    "docling_available": self.docling.available(),
                    "vlm_available": self.vlm.available(self.config),
                },
            }
            self._write_json(root / "routing_manifest.json", manifest)
            if dry_run:
                return {
                    "output_dir": str(root),
                    "pages_profiled": len(profiles),
                    "dry_run": True,
                    "classes": dict(Counter(profile.page_class.value for profile in profiles.values())),
                }

            review_records: List[Dict[str, Any]] = []
            processed: Dict[int, Dict[str, Any]] = {}
            candidates = [
                page_number for page_number in profiles
                if force or not (pages_dir / f"page_{page_number:03d}.json").exists()
            ]
            docling_pages = [
                page_number for page_number in candidates
                if routes[page_number].engine == EngineName.DOCLING
            ]

            # Native pages are processed in contiguous runs to preserve Docling context.
            docling_output: Dict[int, str] = {}
            docling_failures: Dict[int, str] = {}
            if docling_pages and self.docling.available():
                for run_first, run_last in _contiguous_runs(docling_pages):
                    try:
                        docling_output.update(self.docling.parse_run(pdf_path, run_first, run_last))
                    except (EngineUnavailableError, EngineExecutionError) as error:
                        for page_number in range(run_first, run_last + 1):
                            docling_failures[page_number] = str(error)

            for page_number in candidates:
                page = document[page_number - 1]
                profile = profiles[page_number]
                route = routes[page_number]
                markdown = ""
                qc: Optional[QCResult] = None

                if route.engine == EngineName.DOCLING and page_number in docling_output:
                    markdown = sanitize_markdown(docling_output[page_number])
                    qc = evaluate_output(markdown, profile, EngineName.DOCLING, self.config)
                    if qc.status != QCStatus.FAIL:
                        self._write_page(
                            pages_dir=pages_dir, document_id=document_id, page=page,
                            profile=profile, route=route, markdown=markdown, qc=qc,
                        )
                        processed[page_number] = {"engine": route.engine.value, "qc": qc.status.value}
                        continue
                    route = RouteDecision(
                        pdf_page=page_number,
                        engine=self.vlm_engine_name,
                        reason=["docling_qc_failed", *qc.failures],
                        attempt=2,
                        fallback_from=EngineName.DOCLING,
                    )
                elif route.engine == EngineName.DOCLING:
                    route = RouteDecision(
                        pdf_page=page_number,
                        engine=self.vlm_engine_name,
                        reason=["docling_did_not_return_page_output", docling_failures.get(page_number, "")],
                        attempt=2,
                        fallback_from=EngineName.DOCLING,
                    )

                if not self.vlm.available(self.config):
                    api_key_name = "DEEPSEEK_API_KEY" if self.config.vlm_provider == "deepseek" else "GEMINI_API_KEY"
                    review_records.append(self._review_record(
                        profile, route, qc,
                        f"VLM route required but {api_key_name} is not configured",
                    ))
                    processed[page_number] = {"engine": EngineName.MANUAL_REVIEW.value, "qc": "not_run"}
                    continue


                try:
                    markdown, vlm_qc, final_route = self._parse_with_vlm(page=page, profile=profile, route=route)
                    self._write_page(
                        pages_dir=pages_dir, document_id=document_id, page=page,
                        profile=profile, route=final_route, markdown=markdown, qc=vlm_qc,
                    )
                    processed[page_number] = {"engine": final_route.engine.value, "qc": vlm_qc.status.value}
                    if vlm_qc.status == QCStatus.FAIL:
                        review_records.append(self._review_record(profile, final_route, vlm_qc))
                except (EngineUnavailableError, EngineExecutionError) as error:
                    review_records.append(self._review_record(profile, route, qc, str(error)))
                    processed[page_number] = {"engine": EngineName.MANUAL_REVIEW.value, "qc": "not_run"}

            if review_records:
                queue_path = root / "review_queue.jsonl"
                queue_path.write_text(
                    "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in review_records),
                    encoding="utf-8",
                )
            manifest["result"] = {
                "processed": processed,
                "review_queue_count": len(review_records),
                "engine_counts": dict(Counter(item["engine"] for item in processed.values())),
                "qc_counts": dict(Counter(item["qc"] for item in processed.values())),
            }
            self._write_json(root / "routing_manifest.json", manifest)
            return {"output_dir": str(root), **manifest["result"]}
        finally:
            document.close()
