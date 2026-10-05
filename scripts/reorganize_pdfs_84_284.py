import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_eval_84_doc_names() -> set:
    jsonl_path = ROOT_DIR / "data" / "gold_test_set" / "financebench" / "data" / "financebench_open_source.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Cannot find FinanceBench JSONL at: {jsonl_path}")

    eval_docs = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_name = item.get("doc_name")
            if doc_name:
                eval_docs.add(doc_name.strip().lower())
    
    print(f"[1] Loaded {len(eval_docs)} unique doc_names from FinanceBench open source JSONL.")
    return eval_docs

def reorganize_pdf_dir(base_dir: Path, eval_docs: set):
    print(f"\n==========================================")
    print(f"Processing: {base_dir}")
    print(f"==========================================")
    if not base_dir.exists():
        print(f"Skipping {base_dir} (does not exist).")
        return

    eval_dir = base_dir / "eval_84"
    distractors_dir = base_dir / "distractors_284"
    eval_dir.mkdir(parents=True, exist_ok=True)
    distractors_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pdf files in parts
    pdf_files = []
    parts = [base_dir / f"part_{i}" for i in range(1, 5)]
    for part in parts:
        if part.exists():
            for f in part.glob("*.pdf"):
                pdf_files.append(f)

    print(f"Found {len(pdf_files)} PDFs in part_1..part_4.")

    eval_count = 0
    distractor_count = 0

    for pdf in pdf_files:
        stem_lower = pdf.stem.lower()
        if stem_lower in eval_docs:
            target_path = eval_dir / pdf.name
            shutil.move(str(pdf), str(target_path))
            eval_count += 1
        else:
            target_path = distractors_dir / pdf.name
            shutil.move(str(pdf), str(target_path))
            distractor_count += 1

    print(f"Moved {eval_count} files to {eval_dir.name}")
    print(f"Moved {distractor_count} files to {distractors_dir.name}")

    # Verify counts
    actual_eval = len(list(eval_dir.glob("*.pdf")))
    actual_distractor = len(list(distractors_dir.glob("*.pdf")))
    print(f"Verification: {actual_eval} in eval_84, {actual_distractor} in distractors_284.")
    assert actual_eval == 84, f"Expected 84 in eval_84, got {actual_eval}"
    assert actual_distractor == 284, f"Expected 284 in distractors_284, got {actual_distractor}"

    # Clean up empty part directories
    for part in parts:
        if part.exists():
            remaining_in_part = list(part.iterdir())
            if len(remaining_in_part) == 0:
                part.rmdir()
                print(f"Removed empty folder: {part.name}")
            else:
                print(f"Warning: {part.name} still contains {len(remaining_in_part)} non-pdf items.")

    # Create manifest
    manifest = {
        "description": "SEC filings partitioned into 84 evaluation targets (FinanceBench) and 284 distractor documents.",
        "eval_84": {
            "total_files": actual_eval,
            "files": sorted([f.name for f in eval_dir.glob("*.pdf")])
        },
        "distractors_284": {
            "total_files": actual_distractor,
            "files": sorted([f.name for f in distractors_dir.glob("*.pdf")])
        }
    }
    manifest_path = base_dir / "partition_manifest_84_284.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote partition manifest to: {manifest_path.name}")

def main():
    eval_docs = load_eval_84_doc_names()
    assert len(eval_docs) == 84, f"Expected 84 unique documents, got {len(eval_docs)}"

    # 1. Reorganize data/sec_filings/pdfs
    sec_filings_dir = ROOT_DIR / "data" / "sec_filings" / "pdfs"
    reorganize_pdf_dir(sec_filings_dir, eval_docs)

    # 2. Reorganize data/gold_test_set/financebench/pdfs
    fb_dir = ROOT_DIR / "data" / "gold_test_set" / "financebench" / "pdfs"
    reorganize_pdf_dir(fb_dir, eval_docs)

    print("\n[SUCCESS] Reorganization of 84/284 files complete across both directories!")

if __name__ == "__main__":
    main()
