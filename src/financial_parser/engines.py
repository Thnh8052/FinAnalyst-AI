"""Parser architecture following Factory Pattern for Document AI.

Includes:
- BaseParser: Abstract base interface for all engines.
- DoclingParser: Vector-native PDF + CUDA TableFormer GPU acceleration.
- VLMPdfParser: Vision LLM adapter supporting DeepSeek Vision & Google Gemini.
- TATRTableParser: Table Transformer offline table structure recognition.
- ParserFactory: Central factory to instantiate parser engines.
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import ParserConfig
from .markdown_utils import clean_markdown_table_pipes
from .models import EngineName


class EngineUnavailableError(RuntimeError):
    """Raised when a route cannot run because its optional dependency is absent."""


class EngineExecutionError(RuntimeError):
    """Raised when an engine returned no usable result."""


class BaseParser(ABC):
    """Abstract base class for all financial document parsers."""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    @abstractmethod
    def is_available(self) -> bool:
        """Check if parser dependencies, GPU, or API keys are available."""
        pass

    @abstractmethod
    def parse_page(self, page: Any, pdf_page: int) -> str:
        """Parse a single PDF page into faithful GitHub-flavored Markdown."""
        pass

    def parse_run(self, pdf_path: Path, first_page: int, last_page: int) -> Dict[int, str]:
        """Batch-parse a contiguous run of pages (optional for batch engines)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support multi-page batch runs.")


class DoclingParser(BaseParser):
    """Batch native-text parser with GPU TableFormer acceleration."""

    def __init__(self, config: ParserConfig) -> None:
        super().__init__(config)
        self._converter: Any = None

    def is_available(self) -> bool:
        return self.available()

    @staticmethod
    def available() -> bool:
        try:
            import docling  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as error:
            raise EngineUnavailableError("Docling is not installed. Install requirements-parsing.txt.") from error

        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = True
        options.table_structure_options.mode = TableFormerMode.ACCURATE

        # Enable GPU acceleration (CUDA)
        try:
            import torch

            if torch.cuda.is_available() and hasattr(options, "accelerator_options"):
                options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA,
                    num_threads=4,
                )
            elif hasattr(options, "accelerator_options"):
                options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.AUTO,
                    num_threads=1,
                )
        except Exception:
            pass

        if hasattr(options, "document_timeout"):
            options.document_timeout = self.config.docling_timeout_seconds
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        return self._converter

    @staticmethod
    def _items_for_page(docling_document: Any, page_number: int) -> str:
        pieces: List[str] = []
        for item, level in docling_document.iterate_items():
            provenance = getattr(item, "prov", None)
            if not provenance or getattr(provenance[0], "page_no", None) != page_number:
                continue
            item_type = type(item).__name__
            if item_type == "TableItem":
                try:
                    content = item.export_to_markdown(doc=docling_document)
                except TypeError:
                    content = item.export_to_markdown()
                if content:
                    content = clean_markdown_table_pipes(content)
            elif item_type == "SectionHeaderItem":
                content = f"{'#' * min(max(int(getattr(item, 'level', 1)), 1), 6)} {getattr(item, 'text', '')}"
            else:
                content = getattr(item, "text", "")
            if content and str(content).strip():
                pieces.append(str(content).strip())
        return "\n\n".join(pieces)

    def parse_run(self, pdf_path: Path, first_page: int, last_page: int) -> Dict[int, str]:
        """Convert one contiguous, one-indexed page range and retain provenance."""
        converter = self._get_converter()
        try:
            result = converter.convert(str(pdf_path), page_range=(first_page, last_page))
        except Exception as error:
            raise EngineExecutionError(f"Docling conversion failed for pages {first_page}-{last_page}: {error}") from error

        parsed: Dict[int, str] = {}
        for page_number in range(first_page, last_page + 1):
            value = self._items_for_page(result.document, page_number)
            if value:
                parsed[page_number] = value
        return parsed

    def parse_page(self, page: Any, pdf_page: int) -> str:
        """Parse a single page via Docling."""
        pdf_path = Path(page.parent.name)
        results = self.parse_run(pdf_path, pdf_page, pdf_page)
        return results.get(pdf_page, "")


class DeepSeekVisionAdapter:
    """Vision adapter communicating with DeepSeek OpenAI-compatible chat endpoint."""

    SYSTEM_PROMPT = """You transcribe one page from a financial report (SEC Form 10-K, IFRS, or VAS).
Return only faithful GitHub-flavored Markdown. Do not summarize, calculate, infer, translate, or correct values.
Transcribe all visible labels, footnotes, units, reporting periods, and table cells.
Preserve a nil hyphen ('-'), zero ('0'), blank cells, parentheses for negative values, and original wording.
Use a Markdown pipe table when the source contains a table. For multi-level headers, make each output column header explicit.
Ignore only non-data decorative stamps/signatures. Include `<!-- PRINTED_PAGE: N -->` only when N is visibly printed.
"""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    @staticmethod
    def available(config: ParserConfig) -> bool:
        return bool(config.deepseek_api_key)

    def parse_page(self, page: Any, pdf_page: int) -> str:
        if not self.config.deepseek_api_key:
            raise EngineUnavailableError("DEEPSEEK_API_KEY is required for DeepSeek VLM-routed pages.")
        try:
            import requests
        except ImportError as error:
            raise EngineUnavailableError("requests is not installed. Install requirements-parsing.txt.") from error

        try:
            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
            image_bytes = pixmap.tobytes("jpeg")
        except Exception as error:
            raise EngineExecutionError(f"Could not render PDF page {pdf_page}: {error}") from error

        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        endpoint = f"{self.config.deepseek_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.deepseek_model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Transcribe PDF page index: {pdf_page} into faithful Markdown with pipe tables."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.config.vlm_max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
        }
        last_error = "unknown API error"
        for attempt in range(1, 4):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            except requests.RequestException as error:
                last_error = str(error)
            else:
                if response.status_code == 200:
                    body = response.json()
                    choices = body.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        if content:
                            return content
                    last_error = "API response contains no text choice"
                elif response.status_code not in (429, 500, 502, 503, 504):
                    raise EngineExecutionError(f"DeepSeek returned HTTP {response.status_code}: {response.text[:500]}")
                else:
                    last_error = f"DeepSeek returned HTTP {response.status_code}"
            time.sleep(2 ** attempt)
        raise EngineExecutionError(f"DeepSeek failed for PDF page {pdf_page}: {last_error}")


class GeminiVisionAdapter:
    """Vision adapter communicating with Google Gemini generateContent endpoint."""

    SYSTEM_PROMPT = """You transcribe one page from a financial report (VAS, IFRS, or SEC filing).
Return only faithful GitHub-flavored Markdown. Do not summarize, calculate, infer, translate, or correct values.
Transcribe all visible labels, footnotes, units, reporting periods, and table cells.
Preserve a nil hyphen ('-'), zero ('0'), blank cells, parentheses for negative values, and the original language.
Use a Markdown pipe table when the source contains a table. For multi-level headers, make each output column header explicit.
Ignore only non-data decorative stamps/signatures. Include `<!-- PRINTED_PAGE: N -->` only when N is visibly printed.
"""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    @staticmethod
    def available(config: ParserConfig) -> bool:
        return bool(config.gemini_api_key)

    def parse_page(self, page: Any, pdf_page: int) -> str:
        if not self.config.gemini_api_key:
            raise EngineUnavailableError("GEMINI_API_KEY is required for VLM-routed pages.")
        try:
            import requests
        except ImportError as error:
            raise EngineUnavailableError("requests is not installed. Install requirements-parsing.txt.") from error

        try:
            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
            image_bytes = pixmap.tobytes("jpeg")
        except Exception as error:
            raise EngineExecutionError(f"Could not render PDF page {pdf_page}: {error}") from error

        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": f"{self.SYSTEM_PROMPT}\n\nPDF page index: {pdf_page}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 16384},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.config.gemini_api_key}
        last_error = "unknown API error"
        for attempt in range(1, 4):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            except requests.RequestException as error:
                last_error = str(error)
            else:
                if response.status_code == 200:
                    body = response.json()
                    parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    content = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
                    if content:
                        return content
                    last_error = "API response contains no text candidate"
                elif response.status_code not in (429, 500, 502, 503, 504):
                    raise EngineExecutionError(f"Gemini returned HTTP {response.status_code}: {response.text[:500]}")
                else:
                    last_error = f"Gemini returned HTTP {response.status_code}"
            time.sleep(2 ** attempt)
        raise EngineExecutionError(f"Gemini failed for PDF page {pdf_page}: {last_error}")


class LocalHFVisionAdapter:
    """Vision adapter communicating with Local HF VLM with NF4 quantization."""

    SYSTEM_PROMPT = """You transcribe one page from a financial report (VAS, IFRS, or SEC filing).
Return only faithful GitHub-flavored Markdown. Do not summarize, calculate, infer, translate, or correct values.
Transcribe all visible labels, footnotes, units, reporting periods, and table cells.
Preserve a nil hyphen ('-'), zero ('0'), blank cells, parentheses for negative values, and the original language.
Use a Markdown pipe table when the source contains a table. For multi-level headers, make each output column header explicit.
Ignore only non-data decorative stamps/signatures. Include `<!-- PRINTED_PAGE: N -->` only when N is visibly printed.
"""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    @staticmethod
    def available(config: ParserConfig) -> bool:
        return True

    def parse_page(self, page: Any, pdf_page: int) -> str:
        try:
            pixmap = page.get_pixmap(dpi=self.config.render_dpi, alpha=False)
            image_bytes = pixmap.tobytes("png")
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
        except Exception as error:
            raise EngineExecutionError(f"Could not render PDF page {pdf_page}: {error}") from error

        from local_hf_vlm import local_vlm
            
        full_prompt = f"{self.SYSTEM_PROMPT}\n\nPDF page index: {pdf_page}"
        try:
            content = local_vlm.generate_from_b64(image_base64, full_prompt)
            if not content:
                raise EngineExecutionError("Local VLM returned empty content.")
            return content
        except Exception as error:
            raise EngineExecutionError(f"Local VLM failed on page {pdf_page}: {error}") from error


class LlamaParseVisionAdapter:
    """Vision adapter communicating with LlamaParse API V2 for complex financial tables."""

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    @staticmethod
    def available(config: ParserConfig) -> bool:
        if not config.llamaparse_api_key:
            return False
        try:
            import llama_parse  # noqa: F401
            import pymupdf  # noqa: F401
            return True
        except ImportError:
            return False

    def parse_batch(self, pdf_path: Path, page_numbers: List[int]) -> Dict[int, str]:
        if not self.config.llamaparse_api_key:
            raise EngineUnavailableError("LLAMA_CLOUD_API_KEY is required for LlamaParse.")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            from llama_parse import LlamaParse
            import pymupdf as fitz
            import tempfile
            import os
        except ImportError as error:
            raise EngineUnavailableError("llama-parse, nest-asyncio, and pymupdf are required.") from error

        sub_batch_size = 10
        chunks = [page_numbers[i:i + sub_batch_size] for i in range(0, len(page_numbers), sub_batch_size)]
        parsed: Dict[int, str] = {}

        parser = LlamaParse(
            api_key=self.config.llamaparse_api_key,
            result_type="markdown",
            tier="agentic",
            version="latest",
            verbose=False,
        )

        for chunk_idx, chunk_pages in enumerate(chunks, start=1):
            if len(chunks) > 1:
                print(f"      [LlamaParse] Processing sub-batch {chunk_idx}/{len(chunks)} ({len(chunk_pages)} pages)...", flush=True)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                temp_pdf_path = tmp_file.name

            try:
                doc = fitz.open(pdf_path)
                new_doc = fitz.open()
                for p in chunk_pages:
                    new_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)
                new_doc.save(temp_pdf_path)
                new_doc.close()
                doc.close()

                documents = parser.load_data(temp_pdf_path)

                if not documents:
                    raise EngineExecutionError(f"LlamaParse returned no pages for sub-batch {chunk_pages}.")

                for i, p_num in enumerate(chunk_pages):
                    if i < len(documents):
                        parsed[p_num] = documents[i].text
            except Exception as error:
                raise EngineExecutionError(f"LlamaParse batch execution failed: {error}") from error
            finally:
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        return parsed

    def parse_page(self, page: Any, pdf_page: int) -> str:
        pdf_path = Path(page.parent.name)
        results = self.parse_batch(pdf_path, [pdf_page])
        return results.get(pdf_page, "")


class VLMPdfParser(BaseParser):
    """Unified Vision LLM parser supporting LlamaParse, DeepSeek, Gemini, and Local adapters."""

    def __init__(self, config: ParserConfig) -> None:
        super().__init__(config)
        self.adapters = []
        
        # Add adapters in order of priority based on config
        if config.vlm_provider in ("llamaparse", "auto"):
            self.adapters.append(LlamaParseVisionAdapter(config))
        if config.vlm_provider in ("deepseek", "auto"):
            self.adapters.append(DeepSeekVisionAdapter(config))
        if config.vlm_provider in ("gemini", "auto"):
            self.adapters.append(GeminiVisionAdapter(config))
        if config.vlm_provider in ("local", "auto"):
            self.adapters.append(LocalHFVisionAdapter(config))

    def is_available(self) -> bool:
        return any(adapter.available(self.config) for adapter in self.adapters)

    @classmethod
    def available(cls, config: ParserConfig) -> bool:
        return True # Handled dynamically by instance

    def parse_batch_if_supported(self, pdf_path: Path, page_numbers: List[int]) -> Dict[int, str]:
        # Try all adapters that support parse_batch
        for adapter in self.adapters:
            if hasattr(adapter, "parse_batch") and adapter.available(self.config):
                try:
                    return adapter.parse_batch(pdf_path, page_numbers)
                except Exception as e:
                    print(f"      [VLM Batch] Adapter {adapter.__class__.__name__} failed: {e}")
                    continue
        raise NotImplementedError("Batch processing failed or not supported by active adapters.")

    def parse_page(self, page: Any, pdf_page: int) -> str:
        last_error = None
        for adapter in self.adapters:
            if not adapter.available(self.config):
                continue
            try:
                return adapter.parse_page(page, pdf_page)
            except Exception as e:
                last_error = e
                print(f"      [VLM Auto] Adapter {adapter.__class__.__name__} failed for page {pdf_page}. Fallback to next. Error: {e}")
        raise EngineExecutionError(f"All configured VLM providers failed for page {pdf_page}. Last error: {last_error}")


class TATRTableParser(BaseParser):
    """Table Transformer (TATR) parser for local/on-premise offline table recognition."""

    def __init__(self, config: ParserConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._processor: Any = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "microsoft/table-transformer-structure-recognition"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = TableTransformerForObjectDetection.from_pretrained(model_id).to(device)
            self._model.eval()
        except Exception as error:
            raise EngineExecutionError(f"Failed to load TATR model: {error}") from error

    def parse_page(self, page: Any, pdf_page: int) -> str:
        if not self.is_available():
            raise EngineUnavailableError("TATR dependencies not installed. Install transformers, torch, and timm.")
        try:
            self._load_model()
        except EngineExecutionError as err:
            raise EngineExecutionError(f"TATR offline model weights not available: {err}") from err

        try:
            import io
            import torch
            from PIL import Image

            pix = page.get_pixmap(dpi=self.config.render_dpi)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            device = next(self._model.parameters()).device
            inputs = self._processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = self._processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

            if len(results["boxes"]) == 0:
                return page.get_text()
            return page.get_text()
        except Exception as error:
            raise EngineExecutionError(f"TATR execution error on page {pdf_page}: {error}") from error


class ParserFactory:
    """Central factory creating document parser instances based on engine contract."""

    @staticmethod
    def create_parser(engine: EngineName | str, config: ParserConfig) -> BaseParser:
        engine_enum = EngineName(engine) if isinstance(engine, str) else engine
        if engine_enum == EngineName.DOCLING:
            return DoclingParser(config)
        elif engine_enum in (EngineName.DEEPSEEK_VLM, EngineName.GEMINI_VLM, EngineName.LOCAL_VLM):
            return VLMPdfParser(config)
        elif engine_enum == EngineName.TATR:
            return TATRTableParser(config)
        raise ValueError(f"Unsupported parser engine: {engine}")

    @staticmethod
    def create_vlm_parser(config: ParserConfig) -> VLMPdfParser:
        return VLMPdfParser(config)


# Backward-compatibility aliases
DoclingEngine = DoclingParser
DeepSeekVisionEngine = VLMPdfParser
GeminiVisionEngine = VLMPdfParser
create_vlm_engine = ParserFactory.create_vlm_parser

