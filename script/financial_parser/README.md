# Financial Parser

Maintained parser for Vietnamese financial statements (VAS), IFRS reports and SEC filings (10-K/10-Q).

It keeps the legacy scripts untouched and provides one auditable flow:

```text
PDF → page profiler → Docling native-text batch or Gemini VLM → post-parse QC
    → canonical page JSON → Structure Chunking
```

## Installation

Use a project virtual environment, then install the parser runtime:

```powershell
python -m pip install -r requirements-parsing.txt
```

Set `GEMINI_API_KEY` only when pages may need the VLM route. Native-text pages can start with Docling without it; scan/OCR-layer pages without a key are written to `review_queue.jsonl` instead of silently failing.

Optional variables:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
FINANCIAL_PARSER_DPI=280
```

## Commands

```powershell
# Inspect page classes and proposed routes. No model/API call.
python script/parse_financial_reports.py data/MWG_BCTC_2025.pdf --dry-run

# Parse a Vietnamese BCTC or 10-K. Output has manifest, page Markdown and canonical JSON.
python script/parse_financial_reports.py data/MWG_BCTC_2025.pdf --out output/mwg_2025
python script/parse_financial_reports.py data/NVIDIA-2025-Annual-Report.pdf --out output/nvidia_2025

# Debug a small page range, without external VLM calls.
python script/parse_financial_reports.py data/MWG_BCTC_2025.pdf --start 30 --end 35 --no-vlm
```

## Output contract

```text
output/<document>/
├── routing_manifest.json   # profiles, initial routes, runtime and run summary
├── review_queue.jsonl      # pages that need a human after a failed route/QC
└── pages/
    ├── page_001.md         # faithful parser Markdown
    └── page_001.json       # canonical contract for Structure Chunking
```

Structure Chunking must treat `page_*.json` as source-of-truth and index only pages whose `page.qc.status` is `pass` or explicitly approved `warning`. Every table block includes `table_id`, section resolution, unit/period evidence, parser route and PDF-page provenance.

## Debugging order

1. Check `routing_manifest.json`: wrong physical page class is a profiler problem.
2. Inspect `page_XXX.md`: wrong cells/tables are an engine or prompt problem.
3. Inspect `page_XXX.json`: lost unit, period, heading or table schema is a canonicalization problem.
4. Read `review_queue.jsonl`: never ingest its failures automatically.
