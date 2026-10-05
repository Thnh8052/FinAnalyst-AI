"""
FinanceBench End-to-End Generation with NVIDIA Nemotron 3 Ultra
==============================================================
Runs Phase 2:
1. Loads 150 FinanceBench questions.
2. Retrieves Top-k chunks from Vector Stores (Single Store or Shared Store).
3. Calls OpenRouter API (nvidia/nemotron-3-ultra-550b-a55b:free) to answer questions.
4. Auto-evaluates responses into Correct / Incorrect / Refusal labels.
5. Saves results incrementally to JSONL.
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

from typing import List, Dict, Any, Tuple
import re
import json
import time
import argparse
import requests
import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "data" / "gold_test_set" / "financebench" / "data" / "financebench_open_source.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"


def normalize_doc_name(doc_name: str) -> str:
    return doc_name.lower().replace("-", "_").strip()


def load_gold_questions():
    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_vector_store(store_dir: Path):
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


def is_chunk_hit(chunk, target_doc_id, target_pages_1indexed, evidence_texts):
    chunk_doc_id = chunk.get("doc_id", "")
    chunk_page = chunk.get("page_num", -1)
    if chunk_doc_id and target_doc_id and chunk_doc_id != target_doc_id:
        return False
    if chunk_page in target_pages_1indexed:
        return True
    chunk_text = chunk.get("text", "")
    for ev_text in evidence_texts:
        if ev_text:
            cleaned_ev = "".join(ev_text.split())[:60]
            cleaned_chk = "".join(chunk_text.split())
            if len(cleaned_ev) >= 30 and cleaned_ev in cleaned_chk:
                return True
    return False


def call_nemotron(prompt: str, max_retries: int = 4) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FinAnalyst-AI",
        "X-Title": "FinAnalyst-AI Benchmark"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and knowledgeable financial analyst answering questions based on SEC filings."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.01,
        "max_tokens": 800
    }

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=35
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                print(f"   [RATE LIMIT] 429 encountered, sleeping {delay:.1f}s...", flush=True)
                time.sleep(delay)
                delay *= 2
            else:
                print(f"   [API ERROR {resp.status_code}] {resp.text[:200]}", flush=True)
                time.sleep(delay)
                delay *= 1.5
        except Exception as e:
            print(f"   [NETWORK ERROR] {e}, retrying...", flush=True)
            time.sleep(delay)
            delay *= 1.5

    return "Error: Unable to get response from API."


def classify_answer(model_answer: str, gold_answer: Any) -> str:
    """Classify answer into Correct Answer, Refusal, or Incorrect Answer with correct precedence."""
    text_lower = model_answer.lower()
    gold_str = str(gold_answer).strip().lower()

    # 1. PRIORITY CHECK: Numerical Match
    try:
        # Normalize gold number string: remove $, %, commas, spaces
        cleaned_gold = gold_str.replace("$", "").replace(",", "").replace("%", "").strip()
        gold_num = float(cleaned_gold)
        # Extract all numbers from model answer
        numbers = re.findall(r"[-+]?\d*\.?\d+", model_answer.replace(",", ""))
        for num_s in numbers:
            try:
                val = float(num_s)
                # Check absolute difference <= 0.015 or relative difference <= 2%
                if abs(val - gold_num) < 0.015 or (gold_num != 0 and abs(val - gold_num) / abs(gold_num) < 0.025):
                    return "Correct Answer"
                # If gold was ratio e.g. 0.042 and model output 4.2% (val = 4.2)
                if abs(val / 100.0 - gold_num) < 0.005 or (gold_num != 0 and abs(val / 100.0 - gold_num) / abs(gold_num) < 0.025):
                    return "Correct Answer"
                # If gold was percent e.g. 4.2% and model output 0.042
                if abs(val * 100.0 - gold_num) < 0.5 or (gold_num != 0 and abs(val * 100.0 - gold_num) / abs(gold_num) < 0.025):
                    return "Correct Answer"
            except ValueError:
                continue
    except ValueError:
        pass

    # 2. PRIORITY CHECK: Exact or substring match for text answers
    if len(gold_str) > 3 and gold_str in text_lower:
        return "Correct Answer"

    # 3. PRIORITY CHECK: Semantic / Keyword overlap for descriptive gold answers
    gold_words = [w for w in re.findall(r"\w+", gold_str) if len(w) > 2 and w not in {"the", "and", "for", "that", "this", "with"}]
    if len(gold_words) >= 2:
        model_words = set(re.findall(r"\w+", text_lower))
        matched = sum(1 for w in gold_words if w in model_words)
        overlap = matched / len(gold_words)
        # If question is yes/no and confirmation matches
        if (gold_str.startswith("yes") and "yes" in text_lower) or (gold_str.startswith("no") and "no" in text_lower):
            if overlap >= 0.35:
                return "Correct Answer"
        elif overlap >= 0.55:
            return "Correct Answer"

    # 4. REFUSAL CHECK (Only if not already matched as Correct)
    refusal_phrases = [
        "cannot answer", "cannot determine", "cannot be determined", "does not contain", "do not contain",
        "does not provide", "do not provide", "not provided", "not mentioned", "not explicitly stated",
        "unable to answer", "unable to determine", "unable to calculate", "insufficient information",
        "not enough information", "don't have enough information", "do not have enough information",
        "doesn't provide enough", "does not provide enough", "not have enough information",
        "does not include", "do not include", "is not included", "are not included",
        "does not disclose", "do not disclose", "not disclose", "no information", "no direct information",
        "cannot be found", "not found in", "not available in", "is not provided", "are not provided",
        "not possible to determine", "cannot make a determination", "i cannot provide", "i'm unable to",
        "as an ai", "don't know", "do not know", "not have access", "excerpts do not", "context does not"
    ]
    if any(phrase in text_lower for phrase in refusal_phrases):
        return "Refusal"

    return "Incorrect Answer"



def run_generation(variant: str = "pymupdf", mode: str = "singleStore", top_k: int = 4, sample_limit: int = None):
    print(f"\n========================================================")
    print(f"[START] Phase 2 Generation: [{variant.upper()}] - Mode: [{mode}]")
    print(f"[MODEL] LLM: {MODEL_NAME}")
    print(f"========================================================")

    if not OPENROUTER_API_KEY:
        raise ValueError("OPEN_ROUTER_API_KEY not found in environment!")

    vs_base = PROJECT_ROOT / "outputs" / "vectorstores"
    if variant.lower() == "pymupdf":
        single_parent = vs_base / "pymupdf_single"
        shared_dir = vs_base / "pymupdf_shared"
    else:
        single_parent = vs_base / "baseline_single"
        shared_dir = vs_base / "baseline_shared"

    questions = load_gold_questions()
    if sample_limit:
        questions = questions[:sample_limit]
    print(f"[INFO] Total questions to evaluate: {len(questions)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"generation_results_{variant}_{mode}.jsonl"
    
    # Load completed IDs if resuming
    completed_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed_ids.add(json.loads(line).get("financebench_id"))
                    except Exception:
                        pass
        print(f"[RESUME] Found {len(completed_ids)} already completed questions.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMBEDDING] Loading BAAI/bge-base-en-v1.5 on [{device.upper()}]...")
    emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    # Load stores
    shared_embs, shared_chunks = None, None
    if mode == "sharedStore":
        print(f"[LOADING] Loading Shared Vector Store...")
        shared_embs, shared_chunks = load_vector_store(shared_dir)

    single_cache = {}

    label_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
    processed = 0

    with open(out_file, "a", encoding="utf-8") as out_f:
        for idx, q in enumerate(questions, 1):
            q_id = q["financebench_id"]
            if q_id in completed_ids:
                continue

            target_doc_id = normalize_doc_name(q["doc_name"])
            ev_pages = [e["evidence_page_num"] + 1 for e in q.get("evidence", []) if "evidence_page_num" in e]
            ev_texts = [e.get("evidence_text", "") for e in q.get("evidence", [])]
            question_text = q["question"]
            gold_answer = q.get("answer")

            # 1. RETRIEVE TOP-K CHUNKS
            q_vec = emb_model.encode([question_text], normalize_embeddings=True, convert_to_numpy=True)[0]
            
            if mode == "singleStore":
                if target_doc_id not in single_cache:
                    single_cache[target_doc_id] = load_vector_store(single_parent / target_doc_id)
                doc_embs, doc_chunks = single_cache[target_doc_id]
                if doc_embs is None or len(doc_chunks) == 0:
                    continue
                scores = np.dot(doc_embs, q_vec)
                top_indices = np.argsort(scores)[::-1][:top_k]
                retrieved_chunks = [doc_chunks[i] for i in top_indices]
            else:
                scores = np.dot(shared_embs, q_vec)
                top_indices = np.argsort(scores)[::-1][:top_k]
                retrieved_chunks = [shared_chunks[i] for i in top_indices]

            # Check if evidence was hit
            hit_evidence = any(is_chunk_hit(c, target_doc_id, ev_pages, ev_texts) for c in retrieved_chunks)
            retrieved_pages = [c.get("page_num") for c in retrieved_chunks]

            # 2. ASSEMBLE CONTEXT PROMPT
            context_blocks = []
            for rank_i, c in enumerate(retrieved_chunks, 1):
                p_num = c.get("page_num", "?")
                d_id = c.get("doc_id", target_doc_id)
                context_blocks.append(f"--- Document Excerpt {rank_i} [Filing: {d_id}, Page: {p_num}] ---\n{c.get('text', '')}")

            context_str = "\n\n".join(context_blocks)
            prompt = (
                f"Use the following pieces of context to answer the question at the end. "
                f"If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                f"{context_str}\n\n"
                f"Question: {question_text}\n"
                f"Helpful Answer:"
            )

            # 3. CALL NEMOTRON VIA OPENROUTER
            print(f"[{idx}/{len(questions)}] Q: {question_text[:70]}... (Hit Ev: {hit_evidence})", flush=True)
            model_ans = call_nemotron(prompt)

            # 4. CLASSIFY
            label = classify_answer(model_ans, gold_answer)
            label_counts[label] += 1
            processed += 1

            print(f"   -> Label: [{label}] | Ev Hit: {hit_evidence}", flush=True)
            print(f"   -> Ans: {model_ans[:100]}...\n", flush=True)

            # 5. WRITE RECORD
            record = {
                "financebench_id": q_id,
                "model_name": MODEL_NAME,
                "eval_mode": mode,
                "variant": variant,
                "doc_name": q["doc_name"],
                "company": q.get("company"),
                "question_type": q.get("question_type"),
                "question": question_text,
                "gold_answer": gold_answer,
                "model_answer": model_ans,
                "evidence_pages_1indexed": ev_pages,
                "retrieved_pages_1indexed": retrieved_pages,
                "hit_evidence": hit_evidence,
                "label": label
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Small rate-limit courtesy pause
            time.sleep(1.8)

    print(f"\n========================================================")
    print(f"[SUMMARY] Mode: {mode} | Processed: {processed}")
    for lbl, cnt in label_counts.items():
        pct = (cnt / max(1, processed)) * 100
        print(f"   - {lbl}: {cnt} ({pct:.1f}%)")
    print(f"[SAVED] Results saved to: {out_file}")
    print(f"========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceBench Phase 2 Generation")
    parser.add_argument("--variant", type=str, default="pymupdf", choices=["pymupdf", "docling"], help="Parsing variant")
    parser.add_argument("--mode", type=str, default="both", choices=["singleStore", "sharedStore", "both"], help="Store mode")
    parser.add_argument("--top_k", type=int, default=4, help="Top k chunks (default: 4)")
    parser.add_argument("--sample", type=int, default=None, help="Sample limit (e.g. 10 for quick test)")
    args = parser.parse_args()

    modes = ["singleStore", "sharedStore"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_generation(variant=args.variant, mode=m, top_k=args.top_k, sample_limit=args.sample)
