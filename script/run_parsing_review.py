"""Run parsing pipeline for NVIDIA 2025 10-K into parsing_review.

Modes:
- Default: Docling-only (--no-vlm) to save tokens across all 130 pages.
- Fallback: Targeted VLM for review queue pages.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "nvidia_2025_10k.pdf"
OUT_DIR = ROOT / "output_test" / "parsing_review"

def run_parsing(no_vlm: bool = True, start: int = 1, end: int = None):
    cmd = [
        sys.executable,
        str(ROOT / "script" / "parse_financial_reports.py"),
        str(PDF_PATH),
        "--out", str(OUT_DIR),
        "--force",
    ]
    if no_vlm:
        cmd.append("--no-vlm")
    if start:
        cmd.extend(["--start", str(start)])
    if end:
        cmd.extend(["--end", str(end)])

    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    no_vlm = "--with-vlm" not in sys.argv
    start_page = 1
    end_page = None
    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            start_page = int(arg.split("=")[1])
        elif arg.startswith("--end="):
            end_page = int(arg.split("=")[1])

    run_parsing(no_vlm=no_vlm, start=start_page, end=end_page)

    # Automatically generate parsing evaluation report in the same folder
    try:
        from script.generate_parsing_evaluation_report import generate_report
        generate_report(OUT_DIR)
    except Exception as err:
        print(f"[WARN] Failed to generate evaluation report automatically: {err}")
