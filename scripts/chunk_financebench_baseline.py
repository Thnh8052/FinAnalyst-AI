"""
FinanceBench Baseline Chunking Pipeline
=======================================
Implements the exact chunking strategy used in the official FinanceBench paper
(Patronus AI & Stanford University):
- Splitter: RecursiveCharacterTextSplitter
- Chunk Size: 1024 characters
- Chunk Overlap: 30 characters
- Input: Docling-parsed pages from outputs/parsing/<doc_id>/pages/page_XXX.json
- Output: outputs/chunking_baseline_1024/<doc_id>/chunks.jsonl

Retains page-level provenance (doc_id, page_num) for exact ground-truth matching.
"""

import os
import sys
import json
import glob
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Standalone fallback if neither is available
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1024, chunk_overlap=30, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if len(text) <= self.chunk_size:
                    return [text] if text.strip() else []
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    if end == len(text):
                        break
                    start = end - self.chunk_overlap
                return chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 30


def get_paths(parser_type: str = "docling"):
    if parser_type.lower() == "pymupdf":
        parsing_dir = PROJECT_ROOT / "outputs" / "parsing_pymupdf"
        output_dir = PROJECT_ROOT / "outputs" / "chunking_pymupdf_1024"
    else:
        parsing_dir = PROJECT_ROOT / "outputs" / "parsing"
        output_dir = PROJECT_ROOT / "outputs" / "chunking_baseline_1024"
    return parsing_dir, output_dir


def chunk_document(doc_id: str, parser_type: str = "docling", force: bool = False) -> Dict[str, Any]:
    """Chunk all pages of a single document using RecursiveCharacterTextSplitter(1024, 30)."""
    parsing_dir, output_dir = get_paths(parser_type)
    doc_parse_dir = parsing_dir / doc_id
    pages_dir = doc_parse_dir / "pages"
    out_doc_dir = output_dir / doc_id
    out_file = out_doc_dir / "chunks.jsonl"
    summary_file = out_doc_dir / "summary.json"

    if not pages_dir.exists():
        return {"status": "missing_pages", "doc_id": doc_id, "chunks": 0}

    if out_file.exists() and not force:
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            if count > 0:
                return {"status": "skipped", "doc_id": doc_id, "chunks": count}
        except Exception:
            pass

    out_doc_dir.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    page_files = sorted(glob.glob(str(pages_dir / "page_*.json")))
    if not page_files:
        # Fallback to page_*.md
        page_md_files = sorted(glob.glob(str(pages_dir / "page_*.md")))
        pages_to_process = [(idx + 1, Path(p).read_text(encoding="utf-8")) for idx, p in enumerate(page_md_files)]
    else:
        pages_to_process = []
        for pf in page_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    pnum = pdata.get("page", {}).get("pdf_page", 1)
                    pmd = pdata.get("page", {}).get("markdown", "")
                    pages_to_process.append((pnum, pmd))
            except Exception as e:
                print(f"  [WARN] Failed to load {pf}: {e}")

    chunks_data = []
    chunk_counter = 0

    for page_num, markdown_text in pages_to_process:
        if not markdown_text or not markdown_text.strip():
            continue

        raw_splits = splitter.split_text(markdown_text)

        for s_idx, split_text in enumerate(raw_splits):
            cleaned = split_text.strip()
            if not cleaned:
                continue

            chunk_counter += 1
            chunk_id = f"{doc_id}_p{page_num:03d}_c{chunk_counter:04d}"

            chunk_item = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "page_num": page_num,
                "source_pages": [page_num],
                "char_count": len(cleaned),
                "text": cleaned,
                "metadata": {
                    "doc_id": doc_id,
                    "page": page_num,
                    "chunker": "recursive_character_1024_30",
                    "baseline": "financebench_paper",
                    "chunk_index": chunk_counter,
                }
            }
            chunks_data.append(chunk_item)

    # Write chunks.jsonl
    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks_data:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    summary = {
        "doc_id": doc_id,
        "total_pages": len(pages_to_process),
        "total_chunks": len(chunks_data),
        "avg_chunk_chars": round(sum(c["char_count"] for c in chunks_data) / max(1, len(chunks_data)), 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"status": "success", "doc_id": doc_id, "chunks": len(chunks_data)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FinanceBench Baseline Chunking (1024/30)")
    parser.add_argument("--parser", type=str, default="docling", choices=["docling", "pymupdf"], help="Input parser engine (docling or pymupdf)")
    parser.add_argument("--doc_id", type=str, default=None, help="Specific doc_id to chunk")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chunk files")
    args = parser.parse_args()

    parsing_dir, output_dir = get_paths(args.parser)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not parsing_dir.exists():
        print(f"❌ Input directory does not exist: {parsing_dir}")
        return

    if args.doc_id:
        res = chunk_document(args.doc_id, parser_type=args.parser, force=args.force)
        print(f"[{res['status'].upper()}] {args.doc_id}: {res['chunks']} chunks created.")
        return

    # Process all parsed directories
    parsed_docs = sorted([d.name for d in parsing_dir.iterdir() if d.is_dir()])
    print(f"\n🚀 Found {len(parsed_docs)} parsed documents in {parsing_dir} (Source: {args.parser.upper()})")
    print(f"⚙️  Settings: RecursiveCharacterTextSplitter (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})\n")

    total_chunks = 0
    success_docs = 0

    for idx, doc_id in enumerate(parsed_docs, 1):
        res = chunk_document(doc_id, parser_type=args.parser, force=args.force)
        print(f"[{idx:02d}/{len(parsed_docs):02d}] {doc_id} -> {res['chunks']} chunks ({res['status']})")
        if res["status"] in ["success", "skipped"]:
            total_chunks += res["chunks"]
            success_docs += 1

    print("\n" + "=" * 60)
    print(f"🎉 Chunking Finished! ({args.parser.upper()} Baseline)")
    print(f"📁 Processed Docs: {success_docs}/{len(parsed_docs)}")
    print(f"📦 Total Baseline Chunks Generated: {total_chunks}")
    print(f"📂 Output Root: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
