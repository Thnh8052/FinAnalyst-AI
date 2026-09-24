"""CLI Orchestrator for Embedding Financial Corpus Chunks per Method and Model.

Executes offline batch embedding across 35 SEC Form 10-K filings for the 5 chunking methods,
storing results in compressed .npz archives without requiring an active vector DB connection.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add 'src' to Python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from financial_rag.config import (
    CHUNK_METHODS,
    COMPANIES,
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    OUTPUT_CHUNKING_ROOT,
    get_embeddings_cache_dir,
    get_provider_config,
)
from financial_rag.embeddings import (
    EmbeddingEngine,
    extract_retrieval_text,
    load_cached_embeddings,
    save_cached_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_embedding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch embedding for SEC 10-K financial chunks."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge_base",
        choices=list(EMBEDDING_PROVIDERS.keys()),
        help=f"Embedding model key (default: 'bge_base'). Options: {list(EMBEDDING_PROVIDERS.keys())}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=list(CHUNK_METHODS.keys()) + ["all"],
        help="Chunking method to embed (default: 'all').",
    )
    parser.add_argument(
        "--company",
        type=str,
        default="all",
        help="Specific company directory name (e.g. 'apple_2024_10k') or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override model inference batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding even if cached .npz file already exists.",
    )
    return parser.parse_args()


def load_chunks_from_jsonl(chunks_file: Path) -> List[Dict[str, Any]]:
    """Read chunks from jsonl file."""
    if not chunks_file.exists():
        return []
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_company_chunks(
    engine: EmbeddingEngine,
    company: str,
    method_key: str,
    method_config: Dict[str, Any],
    cache_dir: Path,
    batch_size: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Embed chunks for a single company and method, saving to .npz cache."""
    cache_path = cache_dir / f"{company}.npz"
    chunks_path = OUTPUT_CHUNKING_ROOT / company / method_config["folder_name"] / "chunks.jsonl"

    if not chunks_path.exists():
        logger.warning(f"[{company}] [{method_key}] Chunks file not found: {chunks_path}")
        return {
            "company": company,
            "chunks_count": 0,
            "status": "missing_source",
            "time_sec": 0.0,
            "speed": 0.0,
            "cache_size_mb": 0.0,
        }

    # Check cache if not force
    if not force and cache_path.exists():
        cached = load_cached_embeddings(cache_path)
        if cached is not None:
            chunk_ids, embs = cached
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            return {
                "company": company,
                "chunks_count": len(chunk_ids),
                "status": "cached",
                "time_sec": 0.0,
                "speed": 0.0,
                "cache_size_mb": cache_size_mb,
            }

    # Load source chunks
    chunks = load_chunks_from_jsonl(chunks_path)
    if not chunks:
        logger.warning(f"[{company}] [{method_key}] No chunks in {chunks_path}")
        return {
            "company": company,
            "chunks_count": 0,
            "status": "empty",
            "time_sec": 0.0,
            "speed": 0.0,
            "cache_size_mb": 0.0,
        }

    # Extract retrieval text
    retrieval_text_key = method_config.get("retrieval_text_key", "content")
    texts = [extract_retrieval_text(c, text_key=retrieval_text_key) for c in chunks]
    chunk_ids = [c.get("chunk_id", f"{company}_{i}") for i, c in enumerate(chunks)]

    # Run embedding
    t0 = time.perf_counter()
    embeddings = engine.encode_texts(texts, batch_size=batch_size, show_progress=False)
    elapsed = time.perf_counter() - t0

    # Save to disk (.npz float16)
    save_cached_embeddings(cache_path, chunk_ids, embeddings)
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
    speed = len(chunks) / elapsed if elapsed > 0 else 0.0

    return {
        "company": company,
        "chunks_count": len(chunks),
        "status": "embedded",
        "time_sec": elapsed,
        "speed": speed,
        "cache_size_mb": cache_size_mb,
    }


def run_embedding_pipeline(
    model_key: str,
    method_keys: List[str],
    companies: List[str],
    batch_size: Optional[int] = None,
    force: bool = False,
) -> None:
    """Run batch embedding pipeline across requested methods and companies."""
    model_cfg = get_provider_config(model_key)
    logger.info("=" * 80)
    logger.info(f"STARTING EMBEDDING PIPELINE: Model='{model_cfg['display_name']}' [{model_key}]")
    logger.info(f"Methods: {len(method_keys)} | Companies: {len(companies)} | Force={force}")
    logger.info("=" * 80)

    # Initialize Engine
    engine = EmbeddingEngine.get_instance(model_key)

    total_pipeline_chunks = 0
    total_pipeline_time = 0.0

    for method_idx, method_key in enumerate(method_keys, 1):
        method_cfg = CHUNK_METHODS[method_key]
        cache_dir = get_embeddings_cache_dir(model_key, method_key)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 80)
        logger.info(
            f"[{method_idx}/{len(method_keys)}] Method: {method_cfg['display_name']} ({method_key})"
        )
        logger.info(f"Retrieval text field: '{method_cfg.get('retrieval_text_key', 'content')}'")
        logger.info(f"Cache directory: {cache_dir}")
        logger.info("-" * 80)

        method_stats = []
        m_t0 = time.perf_counter()

        for c_idx, company in enumerate(companies, 1):
            stat = embed_company_chunks(
                engine=engine,
                company=company,
                method_key=method_key,
                method_config=method_cfg,
                cache_dir=cache_dir,
                batch_size=batch_size,
                force=force,
            )
            method_stats.append(stat)

            status_label = f"[{stat['status'].upper()}]"
            if stat["status"] == "embedded":
                logger.info(
                    f"  ({c_idx:02d}/{len(companies):02d}) {company:<22} "
                    f"-> {stat['chunks_count']:>5} chunks | "
                    f"{stat['time_sec']:>6.2f}s ({stat['speed']:>6.1f} chk/s) | "
                    f"{stat['cache_size_mb']:>5.2f} MB {status_label}"
                )
            else:
                logger.info(
                    f"  ({c_idx:02d}/{len(companies):02d}) {company:<22} "
                    f"-> {stat['chunks_count']:>5} chunks | {status_label}"
                )

            # Periodic memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        m_elapsed = time.perf_counter() - m_t0
        total_chunks = sum(s["chunks_count"] for s in method_stats)
        embedded_chunks = sum(s["chunks_count"] for s in method_stats if s["status"] == "embedded")
        embedded_time = sum(s["time_sec"] for s in method_stats if s["status"] == "embedded")
        total_cache_mb = sum(s["cache_size_mb"] for s in method_stats)
        avg_speed = embedded_chunks / embedded_time if embedded_time > 0 else 0.0

        total_pipeline_chunks += total_chunks
        total_pipeline_time += embedded_time

        # Save Method Summary JSON
        summary_payload = {
            "model_key": model_key,
            "model_name": model_cfg["model_name"],
            "embedding_dim": model_cfg["embedding_dim"],
            "method_key": method_key,
            "method_display_name": method_cfg["display_name"],
            "retrieval_text_key": method_cfg.get("retrieval_text_key", "content"),
            "total_chunks": total_chunks,
            "embedded_chunks_in_run": embedded_chunks,
            "elapsed_inference_seconds": round(embedded_time, 2),
            "wall_clock_seconds": round(m_elapsed, 2),
            "average_speed_chunks_per_sec": round(avg_speed, 1),
            "total_cache_size_mb": round(total_cache_mb, 2),
            "companies_count": len(companies),
        }

        summary_file = cache_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        logger.info(
            f"METHOD COMPLETED: {method_cfg['display_name']} -> "
            f"Total: {total_chunks:,} chunks | "
            f"Inference: {embedded_time:.2f}s ({avg_speed:.1f} chk/s) | "
            f"Cache: {total_cache_mb:.2f} MB | Summary: {summary_file}"
        )

    logger.info("=" * 80)
    logger.info("ALL REQUESTED METHODS EMBEDDED SUCCESSFULLY!")
    logger.info(f"Model: {model_cfg['display_name']} ({model_key})")
    logger.info(f"Total Chunks Processed: {total_pipeline_chunks:,}")
    logger.info(f"Total Inference Time: {total_pipeline_time:.2f}s")
    logger.info("=" * 80)


def main():
    args = parse_args()

    # Determine methods to run (Method 5 prioritized first, followed by baselines)
    if args.method == "all":
        method_keys = [
            "method5_proposed_golden_hybrid",
            "method1_fixed_size",
            "method2_deterministic",
            "method3_heading_llm",
            "method4_boundary_tagging",
        ]
    else:
        method_keys = [args.method]

    # Determine companies to run
    if args.company == "all":
        companies = COMPANIES
    else:
        if args.company not in COMPANIES:
            logger.error(f"Unknown company: '{args.company}'. Available: {COMPANIES}")
            sys.exit(1)
        companies = [args.company]

    run_embedding_pipeline(
        model_key=args.model,
        method_keys=method_keys,
        companies=companies,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
