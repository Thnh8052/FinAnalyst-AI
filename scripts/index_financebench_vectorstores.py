"""
FinanceBench Baseline Vector Indexing
====================================
Builds Vector Stores using BAAI/bge-base-en-v1.5 (or bge-small-en-v1.5):
1. Single Vector Stores: 1 index per document at outputs/vectorstores/baseline_single/<doc_id>/
2. Shared Vector Store: 1 unified index across all documents at outputs/vectorstores/baseline_shared/

Uses sentence-transformers with CUDA acceleration, L2-normalized embeddings,
and exact cosine similarity search (dot product).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_vs_paths(variant: str = "docling"):
    vs_base_dir = PROJECT_ROOT / "outputs" / "vectorstores"
    if variant.lower() == "pymupdf":
        chunk_dir = PROJECT_ROOT / "outputs" / "chunking_pymupdf_1024"
        single_dir = vs_base_dir / "pymupdf_single"
        shared_dir = vs_base_dir / "pymupdf_shared"
    else:
        chunk_dir = PROJECT_ROOT / "outputs" / "chunking_baseline_1024"
        single_dir = vs_base_dir / "baseline_single"
        shared_dir = vs_base_dir / "baseline_shared"
    return chunk_dir, single_dir, shared_dir


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = None):
    from sentence_transformers import SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📦 Loading embedding model: {model_name} on [{device.upper()}]...")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"✅ Model loaded in {time.time() - t0:.2f}s (dim={model.get_sentence_embedding_dimension()})")
    return model


def load_chunks_for_doc(doc_id: str, variant: str = "docling") -> List[Dict[str, Any]]:
    chunk_dir, _, _ = get_vs_paths(variant)
    chunk_file = chunk_dir / doc_id / "chunks.jsonl"
    if not chunk_file.exists():
        return []
    chunks = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_single_store(doc_id: str, model, variant: str = "docling", force: bool = False) -> Dict[str, Any]:
    _, single_dir, _ = get_vs_paths(variant)
    out_dir = single_dir / doc_id
    meta_file = out_dir / "index_meta.json"
    emb_file = out_dir / "embeddings.npy"
    chunks_file = out_dir / "chunks.jsonl"

    if meta_file.exists() and emb_file.exists() and chunks_file.exists() and not force:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"status": "skipped", "doc_id": doc_id, "chunks": meta.get("total_chunks", 0)}

    chunks = load_chunks_for_doc(doc_id, variant=variant)
    if not chunks:
        return {"status": "no_chunks", "doc_id": doc_id, "chunks": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]

    # Batch encode with CUDA
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Save embeddings and chunks
    np.save(str(emb_file), embeddings)
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "doc_id": doc_id,
        "variant": variant,
        "model_name": model._model_card_text if hasattr(model, "_model_card_text") else str(model),
        "total_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}


def build_shared_store(doc_ids: List[str], model, variant: str = "docling", force: bool = False) -> Dict[str, Any]:
    _, _, shared_dir = get_vs_paths(variant)
    shared_dir.mkdir(parents=True, exist_ok=True)
    meta_file = shared_dir / "index_meta.json"
    emb_file = shared_dir / "embeddings.npy"
    chunks_file = shared_dir / "chunks.jsonl"

    if meta_file.exists() and emb_file.exists() and chunks_file.exists() and not force:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"status": "skipped", "total_chunks": meta.get("total_chunks", 0)}

    _, single_dir, shared_dir = get_vs_paths(variant)
    all_chunks = []
    
    # Fast Path: Check if all single stores already have computed embeddings
    all_single_exist = all((single_dir / doc_id / "embeddings.npy").exists() and (single_dir / doc_id / "chunks.jsonl").exists() for doc_id in doc_ids)
    
    if all_single_exist:
        print(f"\n🌐 [Fast Path] Assembling Shared Vector Store by concatenating pre-computed single stores...")
        t0 = time.time()
        sub_embs = [np.load(str(single_dir / doc_id / "embeddings.npy")) for doc_id in doc_ids]
        embeddings = np.concatenate(sub_embs, axis=0)
        for doc_id in doc_ids:
            with open(single_dir / doc_id / "chunks.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_chunks.append(json.loads(line))
        print(f"⚡ Assembled {len(all_chunks)} chunks in {time.time() - t0:.2f}s without duplicate GPU re-encoding!")
    else:
        for doc_id in doc_ids:
            all_chunks.extend(load_chunks_for_doc(doc_id, variant=variant))

        if not all_chunks:
            return {"status": "no_chunks", "total_chunks": 0}

        print(f"\n🌐 Indexing Shared Vector Store with {len(all_chunks)} total chunks across {len(doc_ids)} documents...")
        texts = [c["text"] for c in all_chunks]

        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        encode_time = time.time() - t0
        print(f"⚡ Encoded {len(all_chunks)} chunks in {encode_time:.2f}s ({len(all_chunks)/max(0.1, encode_time):.1f} chunks/sec)")

    np.save(str(emb_file), embeddings)
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "store_type": "shared",
        "total_documents": len(doc_ids),
        "total_chunks": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"status": "success", "total_chunks": len(all_chunks)}


def query_vector_store(
    store_dir: Path,
    query_text: str,
    model,
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """Query a local vector store via exact cosine similarity."""
    emb_file = store_dir / "embeddings.npy"
    chunks_file = store_dir / "chunks.jsonl"

    if not emb_file.exists() or not chunks_file.exists():
        return []

    # BGE recommendation: prefix query with instruction for retrieval
    query_input = f"Represent this sentence for searching relevant passages: {query_text}"
    q_emb = model.encode([query_input], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Load store embeddings
    doc_embeddings = np.load(str(emb_file), mmap_mode="r")

    # Cosine similarity via dot product (both normalized)
    scores = np.dot(doc_embeddings, q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Read chunks
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    results = []
    for idx in top_indices:
        chunk_obj = chunks[idx]
        chunk_obj["score"] = float(scores[idx])
        results.append(chunk_obj)

    return results


def main():
    parser = argparse.ArgumentParser(description="Build Baseline Single and Shared Vector Stores")
    parser.add_argument("--variant", type=str, default="docling", choices=["docling", "pymupdf"], help="Parsing variant (docling or pymupdf)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace embedding model")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Device to run embedding on (cuda or cpu)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing vector indexes")
    parser.add_argument("--skip_shared", action="store_true", help="Only build single stores")
    args = parser.parse_args()

    chunk_dir, single_dir, shared_dir = get_vs_paths(args.variant)

    # Discover chunked documents
    if not chunk_dir.exists():
        print(f"❌ Chunk directory not found: {chunk_dir}. Run scripts/chunk_financebench_baseline.py --parser {args.variant} first.")
        return

    chunked_docs = sorted([d.name for d in chunk_dir.iterdir() if d.is_dir()])
    if not chunked_docs:
        print(f"❌ No chunked documents found in {chunk_dir}. Run scripts/chunk_financebench_baseline.py --parser {args.variant} first.")
        return

    print(f"\n🚀 Found {len(chunked_docs)} chunked documents to index ({args.variant.upper()} variant).")
    model = load_model(args.model, device=args.device)

    # 1. Build Single Stores
    print(f"\n--- 1. Building Single Vector Stores ({len(chunked_docs)} documents) ---")
    indexed_count = 0
    total_single_chunks = 0
    for idx, doc_id in enumerate(chunked_docs, 1):
        res = build_single_store(doc_id, model, variant=args.variant, force=args.force)
        print(f"[{idx:02d}/{len(chunked_docs):02d}] Single Store: {doc_id} -> {res['chunks']} chunks ({res['status']})")
        total_single_chunks += res["chunks"]
        indexed_count += 1

    # 2. Build Shared Store
    if not args.skip_shared:
        print(f"\n--- 2. Building Shared Vector Store ({args.variant.upper()}) ---")
        shared_res = build_shared_store(chunked_docs, model, variant=args.variant, force=args.force)
        print(f"✅ Shared Store: {shared_res['total_chunks']} chunks ({shared_res['status']})")

    print("\n" + "=" * 60)
    print(f"🎉 Vector Indexing Completed! ({args.variant.upper()} Variant)")
    print(f"📁 Single Stores: {single_dir}")
    print(f"📁 Shared Store: {shared_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
