"""
FinanceBench End-to-End Generation & Evaluation with Space Bunny Alpha
=====================================================================
Runs Phase 2 Generation & Evaluation:
1. Loads 150 FinanceBench questions.
2. Retrieves Top-k chunks from Vector Stores (Single Store or Shared Store).
3. Calls OpenRouter API (stealth/space-bunny-alpha) as GENERATOR to answer questions.
4. Calls OpenRouter API (stealth/space-bunny-alpha) as LLM-as-a-JUDGE to grade answers:
   - Correct Answer
   - Incorrect Answer
   - Refusal
5. Saves results incrementally to JSONL for zero-data-loss and seamless resumption.
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

from typing import List, Dict, Any, Tuple, Optional
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "qwen/qwen3.8-27b"
JUDGE_MODEL_NAME = "openai/gpt-oss-20b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}




def normalize_doc_name(doc_name: str) -> str:
    return doc_name.lower().strip()


def get_store_dir(parent_dir: Path, doc_name: str) -> Path:
    p1 = parent_dir / doc_name.lower().strip()
    if (p1 / "embeddings.npy").exists():
        return p1
    p2 = parent_dir / doc_name.lower().replace("-", "_").strip()
    if (p2 / "embeddings.npy").exists():
        return p2
    return p1


def load_gold_questions() -> List[Dict[str, Any]]:
    questions = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_vector_store(store_dir: Path) -> Tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
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
    chunk_doc_id = chunk.get("doc_id", "")
    chunk_page = chunk.get("page_num", -1)
    if chunk_doc_id and target_doc_id:
        if chunk_doc_id.lower().replace("-", "_") != target_doc_id.lower().replace("-", "_"):
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


def call_space_bunny_generator(prompt: str, max_retries: int = 5) -> Tuple[str, str]:
    """Calls DeepSeek API (deepseek-chat) to generate an answer. Returns (model_answer, reasoning)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional financial analyst. Provide a direct, concise, and definitive final answer based solely on the provided excerpts. Do NOT output internal thoughts, reasoning steps, self-corrections, or conversational filler."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1560
    }

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning") or "").strip()
                
                # CRITICAL: Prevent thought process leakage into final model_answer
                if "<think>" in content and "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                elif "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                if not content:
                    content = "The requested financial information is not available in the provided excerpts."
                elif len(content) > 250 and ("Self-Correction" in content or "Let's " in content or "Wait," in content or content.startswith("To calculate") or "we need the" in content.lower()):
                    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                    candidates = [p for p in paragraphs if not (p.startswith("We need") or p.startswith("*Self-Correction") or p.startswith("Let's ") or p.startswith("Wait,") or p.startswith("1.") or p.startswith("2.") or p.startswith("**1.") or p.startswith("**2."))]
                    if candidates:
                        content = candidates[-1]
                return content, reasoning
            elif resp.status_code == 429:
                wait_sec = delay
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_sec = float(retry_after) + 0.5
                    except Exception:
                        pass
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                    m = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if m:
                        wait_sec = max(wait_sec, float(m.group(1)) + 0.5)
                except Exception:
                    pass
                print(f"   [GENERATOR RATE LIMIT 429] Waiting {wait_sec:.1f}s for quota reset...", flush=True)
                time.sleep(wait_sec)
                delay = max(delay * 1.5, wait_sec)
            else:
                print(f"   [API ERROR {resp.status_code}] {resp.text[:200]}", flush=True)
                time.sleep(delay)
                delay *= 1.5
        except Exception as e:
            print(f"   [NETWORK EXCEPTION] {e}, retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
            delay *= 1.5

    return "The requested financial information is not available in the provided excerpts.", ""


def call_space_bunny_judge(question: str, gold_answer: str, justification: str, model_answer: str, max_retries: int = 4) -> Tuple[str, str]:
    """
    Calls Space Bunny Alpha as LLM-as-a-Judge.
    Returns (label, judge_reasoning).
    Label is strictly one of: 'Correct Answer', 'Incorrect Answer', 'Refusal'.
    """
    # Check trivial refusal upfront
    lower_ans = model_answer.lower()
    refusal_starts = [
        "i don't know", "i do not know", "information is not provided",
        "context does not contain", "context does not provide", "unable to answer",
        "cannot answer based on the provided", "excerpts do not contain",
        "the requested financial information is not available"
    ]
    if any(lower_ans.startswith(p) for p in refusal_starts) and len(model_answer.split()) < 35:
        return "Refusal", "Model explicitly stated refusal due to lack of information in context."

    judge_prompt = f"""You are an objective financial benchmark judge evaluating whether a model generated answer is correct compared to the gold standard reference answer.

[QUESTION]: {question}
[GOLD REFERENCE ANSWER]: {gold_answer}
[GOLD JUSTIFICATION / CONTEXT]: {justification if justification else 'N/A'}
[MODEL GENERATED ANSWER]: {model_answer}

Evaluation Rubric:
- 'Correct Answer': The model answer is factually accurate and matches the gold answer. This includes:
  * Minor rounding differences (e.g., 24.3 vs 24.26, or 4.2% vs 4.21%).
  * Unit conversions (e.g., $1,577 million vs $1.577 billion).
  * Semantically equivalent qualitative answers (e.g., 'Yes, it improved' vs 'Yes, the metric rose').
- 'Incorrect Answer': The model attempts an answer but gives contradictory facts, wrong numbers, incorrect trend conclusions, or hallucinated values.
- 'Refusal': The model explicitly states that the context is insufficient, that the required data/financial statements are missing, or that it cannot answer.

Respond strictly in valid JSON format:
{{
  "reasoning": "Concise 1-2 sentence explanation of your judgment",
  "label": "Correct Answer" | "Incorrect Answer" | "Refusal"
}}"""

    payload = {
        "model": JUDGE_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict, objective financial benchmark judge. Output strictly valid JSON."
            },
            {"role": "user", "content": judge_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"}
    }

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                raw_text = (msg.get("content") or "").strip()
                if not raw_text and msg.get("reasoning"):
                    raw_text = msg["reasoning"].strip()

                # Extract JSON block
                json_match = re.search(r"\{[\s\S]*\}", raw_text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        lbl = parsed.get("label", "").strip()
                        rsn = parsed.get("reasoning", "").strip()
                        # Normalize label
                        if "correct" in lbl.lower() and "incorrect" not in lbl.lower():
                            return "Correct Answer", rsn
                        elif "refusal" in lbl.lower():
                            return "Refusal", rsn
                        elif "incorrect" in lbl.lower():
                            return "Incorrect Answer", rsn
                    except Exception:
                        pass
                
                # Direct string check if JSON wasn't parsed
                if '"Correct Answer"' in raw_text or 'label: "Correct Answer"' in raw_text or 'Correct Answer' in raw_text:
                    if 'Incorrect Answer' not in raw_text:
                        return "Correct Answer", raw_text
                if '"Refusal"' in raw_text or 'Refusal' in raw_text:
                    return "Refusal", raw_text
                if '"Incorrect Answer"' in raw_text or 'Incorrect Answer' in raw_text:
                    return "Incorrect Answer", raw_text

                return "Incorrect Answer", raw_text
            elif resp.status_code == 429:
                wait_sec = delay
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_sec = float(retry_after) + 0.5
                    except Exception:
                        pass
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                    m = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if m:
                        wait_sec = max(wait_sec, float(m.group(1)) + 0.5)
                except Exception:
                    pass
                print(f"   [JUDGE RATE LIMIT 429] Waiting {wait_sec:.1f}s for quota reset...", flush=True)
                time.sleep(wait_sec)
                delay = max(delay * 1.5, wait_sec)
            else:
                print(f"   [JUDGE API ERROR {resp.status_code}] {resp.text[:200]}", flush=True)
                time.sleep(delay)
                delay *= 1.5
        except Exception as e:
            print(f"   [JUDGE NETWORK ERROR] {e}, retrying...", flush=True)
            time.sleep(delay)
            delay *= 1.5

    return "Incorrect Answer", "Evaluation failed due to network/API error."


def run_generation(variant: str = "pymupdf", mode: str = "singleStore", top_k: int = 4, sample_limit: Optional[int] = None):
    print(f"\n" + "=" * 65)
    print(f"[START] Phase 2 Generation & Evaluation: [{variant.upper()}] - Mode: [{mode}]")
    print(f"[MODEL] Generator : {MODEL_NAME}")
    print(f"[MODEL] Judge     : {JUDGE_MODEL_NAME}")
    print(f"=" * 65)

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment!")

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
    out_file = OUTPUT_DIR / f"generation_results_spacebunny_{variant}_{mode}.jsonl"

    # Load completed IDs if resuming
    completed_ids = set()
    label_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record.get("financebench_id"))
                        lbl = record.get("label")
                        if lbl in label_counts:
                            label_counts[lbl] += 1
                    except Exception:
                        pass
        print(f"[RESUME] Found {len(completed_ids)} already completed questions. Resuming...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMBEDDING] Loading BAAI/bge-base-en-v1.5 on [{device.upper()}]...")
    emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    # Load stores
    shared_embs, shared_chunks = None, None
    if mode == "sharedStore":
        print(f"[LOADING] Loading Shared Vector Store...")
        shared_embs, shared_chunks = load_vector_store(shared_dir)
        if shared_embs is None:
            raise FileNotFoundError(f"Shared vector store not found in {shared_dir}")

    single_cache = {}
    processed = len(completed_ids)

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
            justification = q.get("justification", "")

            # 1. RETRIEVE TOP-K CHUNKS
            q_vec = emb_model.encode([question_text], normalize_embeddings=True, convert_to_numpy=True)[0]

            if mode == "singleStore":
                if target_doc_id not in single_cache:
                    store_dir = get_store_dir(single_parent, q["doc_name"])
                    single_cache[target_doc_id] = load_vector_store(store_dir)
                doc_embs, doc_chunks = single_cache[target_doc_id]
                if doc_embs is None or len(doc_chunks) == 0:
                    print(f"[{idx}/{len(questions)}] [SKIP] No vector store for {target_doc_id}")
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
                chunk_txt = c.get('text', '')[:2000]
                context_blocks.append(f"--- Document Excerpt {rank_i} [Filing: {d_id}, Page: {p_num}] ---\n{chunk_txt}")

            context_str = "\n\n".join(context_blocks)
            gen_prompt = (
                f"You are a professional financial analyst evaluating SEC filings. Answer the question directly and concisely based ONLY on the excerpts below.\n\n"
                f"STRICT INSTRUCTIONS:\n"
                f"1. Output ONLY the direct final answer in 1-2 sentences. Do NOT output step-by-step calculations, internal thoughts, self-corrections, or intermediate reasoning.\n"
                f"2. If a numerical metric, ratio, or value is requested, state the final number clearly with appropriate units ($, %, million, billion).\n"
                f"3. If the excerpts do not contain the necessary information or financial line items to calculate the answer, state strictly: \"The requested financial information is not available in the provided excerpts.\"\n\n"
                f"{context_str}\n\n"
                f"Question: {question_text}\n"
                f"Final Answer:"
            )

            # 3. GENERATION
            print(f"[{idx}/{len(questions)}] Q: {question_text[:70]}... (Ev Hit: {hit_evidence})", flush=True)
            model_ans, gen_reasoning = call_space_bunny_generator(gen_prompt)

            # 4. LLM-AS-A-JUDGE EVALUATION
            label, judge_reasoning = call_space_bunny_judge(question_text, gold_answer, justification, model_ans)
            label_counts[label] += 1
            processed += 1

            print(f"   -> Verdict: [{label}] | Ev Hit: {hit_evidence}")
            print(f"   -> Model Ans: {model_ans[:90]}...")
            print(f"   -> Judge Rsn: {judge_reasoning[:120]}...\n", flush=True)

            # 5. WRITE RECORD
            record = {
                "financebench_id": q_id,
                "model_name": MODEL_NAME,
                "judge_model": JUDGE_MODEL_NAME,
                "eval_mode": mode,
                "variant": variant,
                "doc_name": q["doc_name"],
                "company": q.get("company"),
                "question_type": q.get("question_type"),
                "question": question_text,
                "gold_answer": gold_answer,
                "justification": justification,
                "model_answer": model_ans,
                "generator_reasoning": gen_reasoning,
                "judge_reasoning": judge_reasoning,
                "evidence_pages_1indexed": ev_pages,
                "retrieved_pages_1indexed": retrieved_pages,
                "hit_evidence": hit_evidence,
                "label": label
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Auto-update HTML & Markdown review files periodically
            if processed % 5 == 0:
                try:
                    from export_manual_review import export_review
                    export_review(out_file)
                except Exception:
                    pass

            # Sleep to respect Groq rate limits comfortably
            time.sleep(2.5)

    print(f"\n" + "=" * 65)
    print(f"[SUMMARY REPORT] Variant: {variant} | Mode: {mode} | Total: {processed}")
    for lbl, cnt in label_counts.items():
        pct = (cnt / max(1, processed)) * 100
        print(f"   - {lbl:<18}: {cnt:>3} ({pct:5.1f}%)")
    print(f"[SAVED] Results saved to: {out_file}")
    print(f"=" * 65 + "\n")

    try:
        from export_manual_review import export_review
        export_review(out_file)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceBench Phase 2 Generation with Space Bunny Alpha")
    parser.add_argument("--variant", type=str, default="pymupdf", choices=["pymupdf", "docling"], help="Parsing variant")
    parser.add_argument("--mode", type=str, default="both", choices=["singleStore", "sharedStore", "both"], help="Store mode")
    parser.add_argument("--top_k", type=int, default=4, help="Top k chunks (default: 4)")
    parser.add_argument("--sample", type=int, default=None, help="Sample limit (e.g. 5 for testing)")
    args = parser.parse_args()

    modes = ["singleStore", "sharedStore"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_generation(variant=args.variant, mode=m, top_k=args.top_k, sample_limit=args.sample)
