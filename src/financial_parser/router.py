"""Orchestrator for VAS/IFRS/SEC parsing with auditable fallback decisions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .canonical import markdown_to_canonical_page
from .config import ParserConfig
from .engines import EngineExecutionError, EngineUnavailableError, ParserFactory
from .markdown_utils import sanitize_markdown
from .models import EngineName, PageClass, PageProfile, QCResult, QCStatus, RouteDecision
from .orphan_recovery import OrphanTextRecoverer
from .profiler import profile_page
from .qc import evaluate_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contiguous_runs(page_numbers: Sequence[int], max_run_size: int = 10) -> List[Tuple[int, int]]:
    """Split page numbers into contiguous runs of at most max_run_size to protect GPU VRAM."""
    if not page_numbers:
        return []
    ordered = sorted(page_numbers)
    runs: List[Tuple[int, int]] = []
    first = previous = ordered[0]
    count = 1
    for current in ordered[1:]:
        if current == previous + 1 and count < max_run_size:
            previous = current
            count += 1
        else:
            runs.append((first, previous))
            first = previous = current
            count = 1
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
        self.docling = ParserFactory.create_parser(EngineName.DOCLING, config)
        self.tatr = ParserFactory.create_parser(EngineName.TATR, config)
        self.vlm = ParserFactory.create_vlm_parser(config)
        
        if config.vlm_provider == "gemini":
            self.vlm_engine_name = EngineName.GEMINI_VLM
        elif config.vlm_provider == "deepseek":
            self.vlm_engine_name = EngineName.DEEPSEEK_VLM
        elif config.vlm_provider == "local":
            self.vlm_engine_name = EngineName.LOCAL_VLM
        else:
            self.vlm_engine_name = EngineName.LLAMAPARSE_VLM
        self.orphan_recoverer = OrphanTextRecoverer()

    def _initial_route(self, profile: PageProfile) -> RouteDecision:
        if profile.page_class == PageClass.NATIVE_TEXT:
            return RouteDecision(profile.pdf_page, EngineName.DOCLING, list(profile.reasons))
        if profile.page_class == PageClass.UNCERTAIN and profile.char_count < 40 and profile.max_image_coverage < 0.05:
            # Blank or near-blank separator page: keep on native path ($0 cost) to avoid wasting VLM tokens
            return RouteDecision(profile.pdf_page, EngineName.DOCLING, list(profile.reasons))
        return RouteDecision(profile.pdf_page, self.vlm_engine_name, list(profile.reasons))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def _is_flawed_page(self, qc: QCResult, route: Optional[RouteDecision] = None) -> bool:
        """Determine if a page has lost cells, warnings, failures, or errors requiring inspection."""
        if qc.status != QCStatus.PASS:
            return True
        if bool(qc.warnings) or bool(qc.failures):
            return True
        if qc.numeric_recall is not None and qc.numeric_recall < self.config.min_docling_numeric_recall:
            return True
        if qc.source_text_recall is not None and qc.source_text_recall < self.config.min_docling_text_recall:
            return True
        if route and (route.attempt > 1 or route.fallback_from is not None):
            return True
        return False

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
        image_path = pages_dir / f"page_{profile.pdf_page:03d}.png"
        md_path.write_text(markdown, encoding="utf-8")

        is_flawed = self._is_flawed_page(qc, route)
        source_image_rel: Optional[str] = None
        if is_flawed:
            try:
                pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
                pixmap.save(str(image_path))
                source_image_rel = str(image_path.relative_to(pages_dir.parent))
            except Exception:
                source_image_rel = None
        elif image_path.exists():
            # If the page passed cleanly, do not keep old flawed image
            try:
                image_path.unlink()
            except OSError:
                pass

        canonical = markdown_to_canonical_page(
            markdown=markdown,
            document_id=document_id,
            pdf_page=profile.pdf_page,
            printed_page=_printed_page(page, markdown),
            profile=profile,
            route=route,
            qc=qc,
        )
        canonical["page"]["is_flawed"] = is_flawed
        canonical["page"]["source_image"] = source_image_rel
        self._write_json(json_path, canonical)

    @staticmethod
    def _continue_with_warning(qc: QCResult) -> QCResult:
        """Keep parsed content usable while preserving every QC failure as a warning."""
        if qc.status != QCStatus.FAIL:
            return qc
        return QCResult(
            status=QCStatus.WARNING,
            source_text_recall=qc.source_text_recall,
            source_text_precision=qc.source_text_precision,
            numeric_recall=qc.numeric_recall,
            numeric_precision=qc.numeric_precision,
            table_shape_pass=qc.table_shape_pass,
            table_count=qc.table_count,
            warnings=[*qc.warnings, *[f"downgraded_from_fail:{item}" for item in qc.failures]],
            failures=[],
        )

    @staticmethod
    def _review_record(
        profile: PageProfile,
        route: RouteDecision,
        qc: Optional[QCResult],
        error: str = "",
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "pdf_page": profile.pdf_page,
            "page_class": profile.page_class.value,
            "route": route.to_dict(),
            "qc": qc.to_dict() if qc else None,
            "error": error or None,
            "source_image": image_path,
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
            import pymupdf as fitz
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
                    "vlm_concurrency": self.config.vlm_concurrency,
                    "render_dpi": self.config.render_dpi,
                    "docling_timeout_seconds": self.config.docling_timeout_seconds,
                },
                "profiles": [profiles[number].to_dict() for number in sorted(profiles)],
                "initial_routes": [routes[number].to_dict() for number in sorted(routes)],
                "runtime": {
                    "docling_available": self.docling.is_available(),
                    "tatr_available": self.tatr.is_available(),
                    "vlm_available": self.vlm.is_available(),
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
            if docling_pages and self.docling.is_available():
                runs = _contiguous_runs(docling_pages)
                for run_idx, (run_first, run_last) in enumerate(runs, 1):
                    print(f"Docling processing batch {run_idx}/{len(runs)}: pages {run_first}-{run_last}...", flush=True)
                    try:
                        docling_output.update(self.docling.parse_run(pdf_path, run_first, run_last))
                    except (EngineUnavailableError, EngineExecutionError) as error:
                        for page_number in range(run_first, run_last + 1):
                            docling_failures[page_number] = str(error)

            # Pass 1: Process Docling outputs and identify VLM candidates
            docling_qc: Dict[int, QCResult] = {}
            for page_number in candidates:
                page = document[page_number - 1]
                profile = profiles[page_number]
                route = routes[page_number]

                if route.engine == EngineName.DOCLING and page_number in docling_output:
                    raw_docling_md = sanitize_markdown(docling_output[page_number])

                    # Recover orphan text / dropped cells / collapsed tables from Docling
                    markdown, recovery_action = self.orphan_recoverer.recover(
                        page=page,
                        markdown=raw_docling_md,
                        raw_text=profile.raw_text,
                        char_count=profile.char_count,
                    )
                    if recovery_action != "untouched":
                        print(f"      [Orphan Recovery] Page {page_number}: {recovery_action}", flush=True)

                    prev_md = (
                        sanitize_markdown(docling_output[page_number - 1])
                        if (page_number - 1) in docling_output
                        else None
                    )
                    next_md = (
                        sanitize_markdown(docling_output[page_number + 1])
                        if (page_number + 1) in docling_output
                        else None
                    )
                    qc = evaluate_output(
                        markdown,
                        profile,
                        EngineName.DOCLING,
                        self.config,
                        prev_page_markdown=prev_md,
                        next_page_markdown=next_md,
                    )
                    if recovery_action != "untouched":
                        qc.warnings.append(f"orphan_recovery_applied:{recovery_action}")
                    docling_qc[page_number] = qc
                    if qc.status != QCStatus.FAIL:
                        self._write_page(
                            pages_dir=pages_dir, document_id=document_id, page=page,
                            profile=profile, route=route, markdown=markdown, qc=qc,
                        )
                        processed[page_number] = {"engine": route.engine.value, "qc": qc.status.value}
                        continue

                    # Pass 1.5: If Docling QC failed on a tabular page, try local offline TATR ($0 offline)
                    if profile.likely_tabular and self.tatr.is_available():
                        try:
                            tatr_md = sanitize_markdown(self.tatr.parse_page(page, page_number))
                            tatr_qc = evaluate_output(tatr_md, profile, EngineName.TATR, self.config)
                            if tatr_qc.status != QCStatus.FAIL:
                                tatr_route = RouteDecision(
                                    pdf_page=page_number,
                                    engine=EngineName.TATR,
                                    reason=["docling_qc_failed_tatr_fallback_passed"],
                                    attempt=2,
                                    fallback_from=EngineName.DOCLING,
                                )
                                self._write_page(
                                    pages_dir=pages_dir,
                                    document_id=document_id,
                                    page=page,
                                    profile=profile,
                                    route=tatr_route,
                                    markdown=tatr_md,
                                    qc=tatr_qc,
                                )
                                processed[page_number] = {"engine": EngineName.TATR.value, "qc": tatr_qc.status.value}
                                continue
                        except Exception:
                            pass

                    # Page is still bad / severely flawed even after recovery & TATR
                    if self.vlm.is_available():
                        # Route to VLM for high-fidelity vision reconstruction
                        routes[page_number] = RouteDecision(
                            pdf_page=page_number,
                            engine=self.vlm_engine_name,
                            reason=["docling_qc_failed", *qc.failures],
                            attempt=2,
                            fallback_from=EngineName.DOCLING,
                        )
                    else:
                        # VLM is disabled or unavailable (--no-vlm mode):
                        # Preserve Docling output with downgraded warnings for downstream testing
                        downgraded_qc = self._continue_with_warning(qc)
                        self._write_page(
                            pages_dir=pages_dir,
                            document_id=document_id,
                            page=page,
                            profile=profile,
                            route=route,
                            markdown=markdown,
                            qc=downgraded_qc,
                        )
                        processed[page_number] = {"engine": route.engine.value, "qc": downgraded_qc.status.value}
                        image_path = pages_dir / f"page_{page_number:03d}.png"
                        rel_img = str(image_path.relative_to(pages_dir.parent)) if image_path.exists() else None
                        review_records.append(self._review_record(
                            profile, route, qc,
                            error="Docling QC failed; VLM fallback unavailable (--no-vlm)",
                            image_path=rel_img,
                        ))
                elif route.engine == EngineName.DOCLING:
                    if self.vlm.is_available():
                        routes[page_number] = RouteDecision(
                            pdf_page=page_number,
                            engine=self.vlm_engine_name,
                            reason=["docling_did_not_return_page_output", docling_failures.get(page_number, "")],
                            attempt=2,
                            fallback_from=EngineName.DOCLING,
                        )
                    else:
                        image_path = pages_dir / f"page_{page_number:03d}.png"
                        try:
                            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
                            pixmap.save(str(image_path))
                            rel_img = str(image_path.relative_to(pages_dir.parent))
                        except Exception:
                            rel_img = None
                        review_records.append(self._review_record(
                            profile, route, None,
                            error=f"Docling did not return page output; VLM unavailable: {docling_failures.get(page_number, '')}",
                            image_path=rel_img,
                        ))
                        processed[page_number] = {"engine": EngineName.MANUAL_REVIEW.value, "qc": "not_run"}

            # Pass 2: Concurrent VLM Processing for remaining pages
            vlm_pages = [p for p in candidates if p not in processed]
            if vlm_pages:
                if not self.vlm.is_available():
                    api_key_name = "LLAMA_CLOUD_API_KEY" if self.config.vlm_provider in ("llamaparse", "auto") else "API_KEY"
                    for page_number in vlm_pages:
                        page = document[page_number - 1]
                        image_path = pages_dir / f"page_{page_number:03d}.png"
                        try:
                            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
                            pixmap.save(str(image_path))
                            rel_img = str(image_path.relative_to(pages_dir.parent))
                        except Exception:
                            rel_img = None
                        review_records.append(self._review_record(
                            profiles[page_number], routes[page_number], docling_qc.get(page_number),
                            f"VLM route required but {api_key_name} is not configured",
                            image_path=rel_img,
                        ))
                        processed[page_number] = {"engine": EngineName.MANUAL_REVIEW.value, "qc": "not_run"}
                else:
                    batch_success = False
                    if hasattr(self.vlm, "parse_batch_if_supported"):
                        try:
                            print(f"VLM processing batch of {len(vlm_pages)} pages...", flush=True)
                            batch_results = self.vlm.parse_batch_if_supported(pdf_path, vlm_pages)
                            for p_num, p_md in batch_results.items():
                                p_page = document[p_num - 1]
                                p_profile = profiles[p_num]
                                p_route = routes[p_num]
                                
                                p_md_sanitized = sanitize_markdown(p_md)
                                p_raw_qc = evaluate_output(p_md_sanitized, p_profile, self.vlm_engine_name, self.config)
                                p_qc = self._continue_with_warning(p_raw_qc)
                                
                                self._write_page(
                                    pages_dir=pages_dir, document_id=document_id, page=p_page,
                                    profile=p_profile, route=p_route, markdown=p_md_sanitized, qc=p_qc,
                                )
                                processed[p_num] = {"engine": p_route.engine.value, "qc": p_qc.status.value}
                                if p_raw_qc.status == QCStatus.FAIL:
                                    p_img = pages_dir / f"page_{p_num:03d}.png"
                                    rel_img = str(p_img.relative_to(pages_dir.parent)) if p_img.exists() else None
                                    review_records.append(self._review_record(p_profile, p_route, p_raw_qc, image_path=rel_img))
                                    
                            batch_success = True
                            vlm_pages = [p for p in vlm_pages if p not in processed]
                        except NotImplementedError:
                            pass # Fallback to concurrent processing
                        except Exception as e:
                            print(f"      [VLM Batch] Batch failed: {e}. Falling back to concurrent page processing.")

                    if vlm_pages:
                        def _process_vlm_worker(p_num: int) -> Tuple[int, Optional[str], Optional[QCResult], Optional[QCResult], RouteDecision, Optional[str]]:
                            p_page = document[p_num - 1]
                            p_profile = profiles[p_num]
                            p_route = routes[p_num]
                            try:
                                p_md, p_raw_qc, p_final = self._parse_with_vlm(page=p_page, profile=p_profile, route=p_route)
                                p_qc = self._continue_with_warning(p_raw_qc)
                                return p_num, p_md, p_qc, p_raw_qc, p_final, None
                            except (EngineUnavailableError, EngineExecutionError) as err:
                                return p_num, None, None, None, p_route, str(err)

                        workers = min(len(vlm_pages), self.config.vlm_concurrency)
                        if workers > 1:
                            with ThreadPoolExecutor(max_workers=workers) as executor:
                                vlm_results = list(executor.map(_process_vlm_worker, vlm_pages))
                        else:
                            vlm_results = [_process_vlm_worker(p) for p in vlm_pages]

                        for p_num, p_md, p_qc, p_raw_qc, p_route, p_err in vlm_results:
                            p_page = document[p_num - 1]
                            p_profile = profiles[p_num]
                            if p_err or p_md is None or p_qc is None:
                                p_image_path = pages_dir / f"page_{p_num:03d}.png"
                                try:
                                    pixmap = p_page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
                                    pixmap.save(str(p_image_path))
                                    rel_img = str(p_image_path.relative_to(pages_dir.parent))
                                except Exception:
                                    rel_img = None
                                review_records.append(self._review_record(
                                    p_profile, p_route, None, p_err or "Empty VLM output", image_path=rel_img
                                ))
                                processed[p_num] = {"engine": EngineName.MANUAL_REVIEW.value, "qc": "not_run"}
                            else:
                                self._write_page(
                                    pages_dir=pages_dir, document_id=document_id, page=p_page,
                                    profile=p_profile, route=p_route, markdown=p_md, qc=p_qc,
                                )
                                processed[p_num] = {"engine": p_route.engine.value, "qc": p_qc.status.value}
                                if p_raw_qc and p_raw_qc.status == QCStatus.FAIL:
                                    p_img = pages_dir / f"page_{p_num:03d}.png"
                                    rel_img = str(p_img.relative_to(pages_dir.parent)) if p_img.exists() else None
                                    review_records.append(self._review_record(p_profile, p_route, p_raw_qc, image_path=rel_img))


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
