"""Reparse 5 target files with the updated orphan_recovery.py and compare Old vs New vs Ground Truth."""

import os
import sys
import glob
import json
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fitz
from financial_parser.config import ParserConfig
from financial_parser.models import EngineName, PageProfile, QCStatus, RouteDecision
from financial_parser.profiler import profile_page
from financial_parser.orphan_recovery import OrphanTextRecoverer
from financial_parser.qc import evaluate_output
from financial_parser.canonical import markdown_to_canonical_page
from financial_parser.markdown_utils import sanitize_markdown

ROOT_DIR = Path(__file__).resolve().parent.parent

def resolve_target_pdf(filename: str) -> str:
    for base in [
        ROOT_DIR / "data" / "sec_filings" / "pdfs",
        ROOT_DIR / "data" / "gold_test_set" / "financebench" / "pdfs",
    ]:
        if base.exists():
            for p in base.rglob("*.pdf"):
                if p.name.lower() == filename.lower():
                    return str(p)
    return str(ROOT_DIR / "data" / "sec_filings" / "pdfs" / "distractors_284" / filename)

TARGET_FILES = [
    {
        "id": "costco_2015_10k",
        "severity": "Mild (5 orphan pages, 3 duplicates)",
        "pdf_path": resolve_target_pdf("COSTCO_2015_10K.pdf"),
    },
    {
        "id": "bestbuy_2016_10k",
        "severity": "Moderate (20 orphan pages, store count continuation, 16 duplicates)",
        "pdf_path": resolve_target_pdf("BESTBUY_2016_10K.pdf"),
    },
    {
        "id": "aes_2016_10k",
        "severity": "Medium-Severe (154 orphan pages, 2 broken table overflows p124 & p176)",
        "pdf_path": resolve_target_pdf("AES_2016_10K.pdf"),
    },
    {
        "id": "boeing_2015_10k",
        "severity": "Severe (48 orphan pages, 37 duplicate pages)",
        "pdf_path": resolve_target_pdf("BOEING_2015_10K.pdf"),
    },
    {
        "id": "amd_2016_10k",
        "severity": "Extreme (76 orphan pages, 75 duplicate pages = 98.7% junk)",
        "pdf_path": resolve_target_pdf("AMD_2016_10K.pdf"),
    },
]

ARCHIVE_ROOT = Path(r"C:\Users\ThanhDz\Downloads\DATN_Finance\FinAnalyst-AI\archive\parsing")
OUTPUT_ROOT = Path(r"C:\Users\ThanhDz\Downloads\DATN_Finance\FinAnalyst-AI\outputs\parsing")

# Exact raw Docling outputs for AES degenerate pages
RAW_DOCLING_SPECIAL = {
    ("aes_2016_10k", 124): """| Total capital expenditures   | $ (2,345)   | $ (2,308)   | $   | (37)   |
|------------------------------|-------------|-------------|-----|--------|

_____________________________

(1) Includes both recoverable and non-recoverable environmental capital expenditures. See SBU Performance Analysis for more information.""",
    ("aes_2016_10k", 176): """| Impairment of assets of equity method investees   | $   | 15   | $   | —   | $   | —   |
|---------------------------------------------------|-----|------|-----|-----|-----|-----|

_____________________________

(1) During 2016, we recognized asset impairment charges of $15 million in continuing operations."""
}


def get_raw_docling_markdown(archive_dir: Path, doc_id: str, page_num: int) -> str:
    """Extract raw Docling markdown from archive by removing old appended orphan junk if present."""
    if (doc_id, page_num) in RAW_DOCLING_SPECIAL:
        return RAW_DOCLING_SPECIAL[(doc_id, page_num)]

    md_file = archive_dir / "pages" / f"page_{page_num:03d}.md"
    if not md_file.exists():
        return ""
    content = md_file.read_text(encoding="utf-8")
    
    # Strip old orphan recovery marker
    marker = "### Ghi chú & Nội dung bổ sung"
    if marker in content:
        content = content.split(marker)[0].rstrip()
    return sanitize_markdown(content)


def reparse_and_evaluate():
    config = ParserConfig()
    orphan_recoverer = OrphanTextRecoverer()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_results = []

    for item in TARGET_FILES:
        doc_id = item["id"]
        pdf_path = Path(item["pdf_path"])
        severity = item["severity"]
        archive_dir = ARCHIVE_ROOT / doc_id
        out_doc_dir = OUTPUT_ROOT / doc_id
        out_pages_dir = out_doc_dir / "pages"
        out_pages_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"Processing: {doc_id} [{severity}]")
        print(f"PDF: {pdf_path.name}")
        print(f"=======================================================")

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        print(f"Total pages: {num_pages}")

        # 1. First pass: Collect all raw docling outputs and profiles
        print("Gathering raw Docling markdown & page profiles...")
        raw_docling_mds: Dict[int, str] = {}
        page_profiles: Dict[int, PageProfile] = {}

        for p_num in range(1, num_pages + 1):
            page = doc[p_num - 1]
            profile = profile_page(page, p_num, config)
            page_profiles[p_num] = profile
            raw_docling_mds[p_num] = get_raw_docling_markdown(archive_dir, doc_id, p_num)

        # 2. Second pass: Run new OrphanTextRecoverer with cross-page context awareness
        print("Running new OrphanTextRecoverer and QC evaluation...")
        
        old_orphan_count = 0
        new_orphan_count = 0
        actions_breakdown: Dict[str, int] = {}
        
        # Metric accumulators
        old_char_total = 0
        new_char_total = 0
        gt_char_total = 0

        old_precisions: List[float] = []
        new_precisions: List[float] = []
        old_recalls: List[float] = []
        new_recalls: List[float] = []

        old_num_precisions: List[float] = []
        new_num_precisions: List[float] = []
        old_num_recalls: List[float] = []
        new_num_recalls: List[float] = []

        fixed_table_continuations = []

        for p_num in range(1, num_pages + 1):
            page = doc[p_num - 1]
            profile = page_profiles[p_num]
            raw_md = raw_docling_mds[p_num]
            prev_md = raw_docling_mds.get(p_num - 1)
            next_md = raw_docling_mds.get(p_num + 1)

            # Check old parse stats from archive
            old_json_path = archive_dir / "pages" / f"page_{p_num:03d}.json"
            old_md_path = archive_dir / "pages" / f"page_{p_num:03d}.md"
            old_md = old_md_path.read_text(encoding="utf-8") if old_md_path.exists() else ""
            old_char_total += len(old_md)
            gt_char_total += profile.char_count

            if old_json_path.exists():
                with open(old_json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                old_qc = old_data.get("page", {}).get("qc", {})
                for w in old_qc.get("warnings", []):
                    if "orphan_recovery_applied" in w:
                        old_orphan_count += 1
                if old_qc.get("source_text_precision") is not None:
                    old_precisions.append(old_qc["source_text_precision"])
                if old_qc.get("source_text_recall") is not None:
                    old_recalls.append(old_qc["source_text_recall"])
                if old_qc.get("numeric_precision") is not None:
                    old_num_precisions.append(old_qc["numeric_precision"])
                if old_qc.get("numeric_recall") is not None:
                    old_num_recalls.append(old_qc["numeric_recall"])

            # RUN NEW ORPHAN RECOVERY
            new_md, action = orphan_recoverer.recover(
                page=page,
                markdown=raw_md,
                raw_text=profile.raw_text,
                char_count=profile.char_count,
                prev_markdown=prev_md,
                next_markdown=next_md,
            )
            new_char_total += len(new_md)
            actions_breakdown[action] = actions_breakdown.get(action, 0) + 1
            if action != "untouched":
                new_orphan_count += 1
                if "reconstructed_table_continuation" in action or "reconstructed_false_table" in action:
                    first_lines = [l for l in new_md.splitlines() if "|" in l][:3]
                    snippet = " \n ".join(first_lines) if first_lines else "N/A"
                    fixed_table_continuations.append({"page": p_num, "action": action, "table_snippet": snippet})

            # Evaluate new QC
            new_qc = evaluate_output(
                new_md,
                profile,
                EngineName.DOCLING,
                config,
                prev_page_markdown=prev_md,
                next_page_markdown=next_md,
            )
            if action != "untouched":
                new_qc.warnings.append(f"orphan_recovery_applied:{action}")

            if new_qc.source_text_precision is not None:
                new_precisions.append(new_qc.source_text_precision)
            if new_qc.source_text_recall is not None:
                new_recalls.append(new_qc.source_text_recall)
            if new_qc.numeric_precision is not None:
                new_num_precisions.append(new_qc.numeric_precision)
            if new_qc.numeric_recall is not None:
                new_num_recalls.append(new_qc.numeric_recall)

            # Write new outputs
            out_md_file = out_pages_dir / f"page_{p_num:03d}.md"
            out_json_file = out_pages_dir / f"page_{p_num:03d}.json"
            out_md_file.write_text(new_md, encoding="utf-8")

            route = RouteDecision(pdf_page=p_num, engine=EngineName.DOCLING, reason=["docling_reparsed"])
            canonical = markdown_to_canonical_page(
                markdown=new_md,
                document_id=doc_id,
                pdf_page=p_num,
                printed_page=None,
                profile=profile,
                route=route,
                qc=new_qc,
            )
            out_json_file.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")

        # Copy manifest / update metadata
        archive_manifest = archive_dir / "routing_manifest.json"
        if archive_manifest.exists():
            with open(archive_manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest_data["orphan_recovery_version"] = "2.0_anchor_and_table_continuation"
            (out_doc_dir / "routing_manifest.json").write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # Calculate comparative metrics
        avg_old_prec = sum(old_precisions) / len(old_precisions) if old_precisions else 0.0
        avg_new_prec = sum(new_precisions) / len(new_precisions) if new_precisions else 0.0
        avg_old_rec = sum(old_recalls) / len(old_recalls) if old_recalls else 0.0
        avg_new_rec = sum(new_recalls) / len(new_recalls) if new_recalls else 0.0

        avg_old_num_prec = sum(old_num_precisions) / len(old_num_precisions) if old_num_precisions else 0.0
        avg_new_num_prec = sum(new_num_precisions) / len(new_num_precisions) if new_num_precisions else 0.0
        avg_old_num_rec = sum(old_num_recalls) / len(old_num_recalls) if old_num_recalls else 0.0
        avg_new_num_rec = sum(new_num_recalls) / len(new_num_recalls) if new_num_recalls else 0.0

        chars_eliminated = old_char_total - new_char_total

        print(f"Results for {doc_id}:")
        print(f"  Orphan Interventions: Old = {old_orphan_count} -> New = {new_orphan_count}")
        print(f"  Actions Breakdown: {actions_breakdown}")
        print(f"  Duplicate Chars Eliminated: {chars_eliminated:,} chars")
        print(f"  Source Text Precision: {avg_old_prec:.4f} -> {avg_new_prec:.4f} (Delta: {avg_new_prec - avg_old_prec:+.4f})")
        print(f"  Source Text Recall:    {avg_old_rec:.4f} -> {avg_new_rec:.4f} (Delta: {avg_new_rec - avg_old_rec:+.4f})")
        print(f"  Numeric Precision:     {avg_old_num_prec:.4f} -> {avg_new_num_prec:.4f} (Delta: {avg_new_num_prec - avg_old_num_prec:+.4f})")
        print(f"  Numeric Recall:        {avg_old_num_rec:.4f} -> {avg_new_num_rec:.4f} (Delta: {avg_new_num_rec - avg_old_num_rec:+.4f})")
        print(f"  Table Continuations Reconstructed: {fixed_table_continuations}")

        summary_results.append({
            "doc_id": doc_id,
            "severity": severity,
            "num_pages": num_pages,
            "old_orphan_pages": old_orphan_count,
            "new_orphan_pages": new_orphan_count,
            "actions": actions_breakdown,
            "chars_eliminated": chars_eliminated,
            "old_text_precision": avg_old_prec,
            "new_text_precision": avg_new_prec,
            "old_text_recall": avg_old_rec,
            "new_text_recall": avg_new_rec,
            "old_num_precision": avg_old_num_prec,
            "new_num_precision": avg_new_num_prec,
            "old_num_recall": avg_old_num_rec,
            "new_num_recall": avg_new_num_rec,
            "fixed_table_continuations": fixed_table_continuations,
        })

    # Save summary report
    summary_path = OUTPUT_ROOT / "comparison_report_5_files.json"
    summary_path.write_text(json.dumps(summary_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Wrote full comparison report to {summary_path}")


if __name__ == "__main__":
    t_start = time.time()
    reparse_and_evaluate()
    print(f"\nAll 5 documents processed in {time.time() - t_start:.2f} seconds.")
