"""
FinanceBench Retrieval Evaluation
=================================
Evaluates 150 benchmark questions on:
1. Single Vector Store (per-document index)
2. Shared Vector Store (cross-document index)

Computes Recall@1, Recall@2, Recall@4, Recall@8, and MRR.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import time
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = PROJECT_ROOT / "data" / "gold_test_set" / "financebench" / "data" / "financebench_open_source.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def normalize_doc_name(doc_name: str) -> str:
    """Normalize SEC filing name to doc_id (e.g. 3M_2018_10K -> 3m_2018_10k)."""
    return doc_name.lower().replace("-", "_").strip()


def load_gold_questions() -> List[Dict[str, Any]]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_vector_store(store_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    emb_file = store_dir / "embeddings.npy"
    chunk_file = store_dir / "chunks.jsonl"
    if not emb_file.exists() or not chunk_file.exists():
        return None, None
    embeddings = np.load(str(emb_file))
    chunks = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return embeddings, chunks


def is_chunk_hit(chunk: Dict[str, Any], target_doc_id: str, target_pages_1indexed: List[int], evidence_texts: List[str]) -> bool:
    """Check if retrieved chunk matches target ground truth evidence."""
    chunk_doc_id = chunk.get("doc_id", "")
    chunk_page = chunk.get("page_num", -1)
    
    # Check document match (in shared store, doc_id must match)
    if chunk_doc_id and target_doc_id and chunk_doc_id != target_doc_id:
        return False
        
    # Check 1-indexed page number match
    if chunk_page in target_pages_1indexed:
        return True
        
    # Check text overlap as secondary check
    chunk_text = chunk.get("text", "")
    for ev_text in evidence_texts:
        if ev_text:
            cleaned_ev = "".join(ev_text.split())[:60]
            cleaned_chk = "".join(chunk_text.split())
            if len(cleaned_ev) >= 30 and cleaned_ev in cleaned_chk:
                return True
                
    return False


def evaluate_variant(variant: str = "pymupdf", device: str = "cuda") -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"[START] Retrieval Evaluation for [{variant.upper()}]")
    print(f"========================================================")
    
    vs_base = PROJECT_ROOT / "outputs" / "vectorstores"
    if variant.lower() == "pymupdf":
        single_parent = vs_base / "pymupdf_single"
        shared_dir = vs_base / "pymupdf_shared"
    else:
        single_parent = vs_base / "baseline_single"
        shared_dir = vs_base / "baseline_shared"
        
    questions = load_gold_questions()
    print(f"[INFO] Loaded {len(questions)} gold benchmark questions.")
    
    print(f"[MODEL] Loading embedding model BAAI/bge-base-en-v1.5 on [{device.upper()}]...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    
    # Pre-encode all 150 questions
    print("[ENCODE] Pre-encoding all 150 questions into embeddings...")
    q_texts = [q["question"] for q in questions]
    q_embs = model.encode(q_texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    
    results = {
        "variant": variant,
        "total_questions": len(questions),
        "single_store": {},
        "shared_store": {},
        "question_details": []
    }
    
    # ----------------------------------------------------
    # 1. EVALUATE SINGLE VECTOR STORE MODE
    # ----------------------------------------------------
    print(f"\n[SINGLE STORE] Evaluating Single Vector Store Mode ({len(questions)} queries)...")
    single_hits_at_k = {1: 0, 2: 0, 4: 0, 8: 0}
    single_mrr_sum = 0.0
    single_cache = {}
    
    for i, q in enumerate(questions):
        target_doc_id = normalize_doc_name(q["doc_name"])
        ev_pages = [e["evidence_page_num"] + 1 for e in q.get("evidence", []) if "evidence_page_num" in e]
        ev_texts = [e.get("evidence_text", "") for e in q.get("evidence", [])]
        
        if target_doc_id not in single_cache:
            store_dir = single_parent / target_doc_id
            single_cache[target_doc_id] = load_vector_store(store_dir)
            
        doc_embs, doc_chunks = single_cache[target_doc_id]
        if doc_embs is None or len(doc_chunks) == 0:
            continue
            
        q_vec = q_embs[i]
        scores = np.dot(doc_embs, q_vec)
        top_k_indices = np.argsort(scores)[::-1][:8]
        
        hit_rank = None
        for rank, idx in enumerate(top_k_indices, 1):
            if is_chunk_hit(doc_chunks[idx], target_doc_id, ev_pages, ev_texts):
                hit_rank = rank
                break
                
        if hit_rank is not None:
            single_mrr_sum += 1.0 / hit_rank
            for k in [1, 2, 4, 8]:
                if hit_rank <= k:
                    single_hits_at_k[k] += 1

    total_q = len(questions)
    results["single_store"] = {
        "recall_at_1": round(single_hits_at_k[1] / total_q * 100, 2),
        "recall_at_2": round(single_hits_at_k[2] / total_q * 100, 2),
        "recall_at_4": round(single_hits_at_k[4] / total_q * 100, 2),
        "recall_at_8": round(single_hits_at_k[8] / total_q * 100, 2),
        "mrr": round(single_mrr_sum / total_q, 4),
        "hits_at_4": single_hits_at_k[4]
    }
    print(f"   Single Store -> Recall@1: {results['single_store']['recall_at_1']}%, Recall@4: {results['single_store']['recall_at_4']}%, MRR: {results['single_store']['mrr']}")

    # ----------------------------------------------------
    # 2. EVALUATE SHARED VECTOR STORE MODE
    # ----------------------------------------------------
    print(f"\n[SHARED STORE] Evaluating Shared Vector Store Mode ({len(questions)} queries)...")
    shared_embs, shared_chunks = load_vector_store(shared_dir)
    shared_hits_at_k = {1: 0, 2: 0, 4: 0, 8: 0}
    shared_mrr_sum = 0.0
    
    if shared_embs is not None:
        for i, q in enumerate(questions):
            target_doc_id = normalize_doc_name(q["doc_name"])
            ev_pages = [e["evidence_page_num"] + 1 for e in q.get("evidence", []) if "evidence_page_num" in e]
            ev_texts = [e.get("evidence_text", "") for e in q.get("evidence", [])]
            
            q_vec = q_embs[i]
            scores = np.dot(shared_embs, q_vec)
            top_k_indices = np.argsort(scores)[::-1][:8]
            
            hit_rank = None
            for rank, idx in enumerate(top_k_indices, 1):
                if is_chunk_hit(shared_chunks[idx], target_doc_id, ev_pages, ev_texts):
                    hit_rank = rank
                    break
                    
            if hit_rank is not None:
                shared_mrr_sum += 1.0 / hit_rank
                for k in [1, 2, 4, 8]:
                    if hit_rank <= k:
                        shared_hits_at_k[k] += 1
                        
            results["question_details"].append({
                "id": q.get("financebench_id"),
                "company": q.get("company"),
                "doc_name": q.get("doc_name"),
                "question_type": q.get("question_type"),
                "question": q.get("question"),
                "evidence_pages_1indexed": ev_pages,
                "shared_hit_rank": hit_rank
            })

    results["shared_store"] = {
        "recall_at_1": round(shared_hits_at_k[1] / total_q * 100, 2),
        "recall_at_2": round(shared_hits_at_k[2] / total_q * 100, 2),
        "recall_at_4": round(shared_hits_at_k[4] / total_q * 100, 2),
        "recall_at_8": round(shared_hits_at_k[8] / total_q * 100, 2),
        "mrr": round(shared_mrr_sum / total_q, 4),
        "hits_at_4": shared_hits_at_k[4]
    }
    print(f"   Shared Store -> Recall@1: {results['shared_store']['recall_at_1']}%, Recall@4: {results['shared_store']['recall_at_4']}%, MRR: {results['shared_store']['mrr']}")

    # ----------------------------------------------------
    # 3. BREAKDOWN BY QUESTION TYPE
    # ----------------------------------------------------
    type_stats = {}
    for item in results["question_details"]:
        q_type = item.get("question_type", "unknown")
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "hits_at_4": 0}
        type_stats[q_type]["total"] += 1
        if item.get("shared_hit_rank") and item["shared_hit_rank"] <= 4:
            type_stats[q_type]["hits_at_4"] += 1

    for q_type, stat in type_stats.items():
        stat["recall_at_4"] = round(stat["hits_at_4"] / max(1, stat["total"]) * 100, 2)
    results["question_type_breakdown"] = type_stats

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"retrieval_evaluation_{variant}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Evaluation results saved to: {out_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinanceBench Retrieval")
    parser.add_argument("--variant", type=str, default="pymupdf", choices=["pymupdf", "docling", "both"], help="Variant to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for embeddings")
    args = parser.parse_args()

    if args.variant in ["pymupdf", "both"]:
        evaluate_variant("pymupdf", device=args.device)
    if args.variant in ["docling", "both"]:
        evaluate_variant("docling", device=args.device)
