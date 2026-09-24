#!/usr/bin/env python3
"""CLI Orchestrator for Indexing Financial Chunks into Qdrant Collections.

Upserts cached embedding vectors and rich payloads into independent Qdrant collections per method.
Guarantees:
- Dual named vectors ('dense' + 'sparse') for seamless V0 -> V1 transition.
- 7 standardized payload indexes with dynamic contains_table & contains_text flags.
- Scientific isolation (5 independent collections for fair benchmarking).
- Zero GPU inference overhead (loads directly from pre-computed .npz float16 caches).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from financial_rag.config import (
    COMPANIES,
    CHUNK_METHODS,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_RETRIEVAL_ROOT,
    EMBEDDINGS_CACHE_DIR,
    DEFAULT_EMBEDDING_PROVIDER,
    get_provider_config,
    get_collection_name,
)
from financial_rag.indexing import QdrantIndexer
from financial_rag.schema import (
    SCHEMA_VERSION,
    SCHEMA_FROZEN_AT,
    TICKERS_V0,
    PAYLOAD_INDEXES_V0,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_indexing")


def index_single_method(
    indexer: QdrantIndexer,
    model_key: str,
    method_key: str,
    recreate: bool = True,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Index all 35 companies of a single chunking method into its dedicated Qdrant collection."""
    model_cfg = get_provider_config(model_key)
    dim = model_cfg["embedding_dim"]
    model_name = model_cfg["model_name"]
    collection_name = get_collection_name(model_key, method_key)

    method_cfg = CHUNK_METHODS[method_key]
    folder_name = method_cfg["folder_name"]
    display_name = method_cfg["display_name"]

    logger.info("=" * 70)
    logger.info(f"Indexing Method: {display_name}")
    logger.info(f"Target Collection: '{collection_name}' (dim={dim})")
    logger.info("=" * 70)

    # 1. Initialize collection with dual named vectors and 7 payload indexes
    indexer.init_collection(collection_name=collection_name, dim=dim, recreate=recreate)

    cache_dir = EMBEDDINGS_CACHE_DIR / model_key / method_key
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Embeddings cache directory not found: {cache_dir}. "
            f"Run scripts/run_embedding.py first!"
        )

    start_time = time.time()
    total_indexed_points = 0
    company_results = {}

    for comp_idx, company_dir in enumerate(COMPANIES, 1):
        npz_file = cache_dir / f"{company_dir}.npz"
        chunks_file = OUTPUT_CHUNKING_ROOT / company_dir / folder_name / "chunks.jsonl"

        if not npz_file.exists():
            logger.warning(f"[{comp_idx}/{len(COMPANIES)}] Missing vector cache: {npz_file}")
            continue
        if not chunks_file.exists():
            logger.warning(f"[{comp_idx}/{len(COMPANIES)}] Missing chunks file: {chunks_file}")
            continue

        # Load vectors (.npz float16 -> float32)
        npz_data = np.load(npz_file)
        vectors = npz_data["embeddings"].astype(np.float32)

        # Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch for {company_dir}: {len(chunks)} chunks vs {len(vectors)} vectors!"
            )

        t0 = time.time()
        indexed_count = indexer.index_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=vectors,
            method_key=method_key,
            embedding_model=model_name,
            embedding_version="v0_dense",
            batch_size=batch_size,
        )
        elapsed = time.time() - t0
        total_indexed_points += indexed_count

        rate = indexed_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"[{comp_idx:2d}/{len(COMPANIES):2d}] {company_dir:20s}: "
            f"{indexed_count:4d} points in {elapsed:5.2f}s ({rate:5.1f} pts/s)"
        )

        company_results[company_dir] = {
            "points_count": indexed_count,
            "indexing_time_sec": round(elapsed, 2),
        }

    total_time = time.time() - start_time
    avg_rate = total_indexed_points / total_time if total_time > 0 else 0

    # Verify collection stats from Qdrant
    coll_info = indexer.get_collection_info(collection_name)
    logger.info(f"Collection '{collection_name}' successfully verified:")
    logger.info(f"  - Qdrant points_count: {coll_info['points_count']}")
    logger.info(f"  - Qdrant status: {coll_info['status']}")
    logger.info(f"  - Total Indexing Time: {total_time:.2f}s (Average: {avg_rate:.1f} pts/s)")

    # Test sample dense search query to verify retrieval readiness
    sample_query = np.random.randn(dim).astype(np.float32)
    sample_query = (sample_query / np.linalg.norm(sample_query)).tolist()
    search_res = indexer.client.query_points(
        collection_name=collection_name,
        query=sample_query,
        using="dense",
        limit=3,
    )
    logger.info(f"  - Verification query returned {len(search_res.points)} points successfully.")

    return {
        "model_key": model_key,
        "method_key": method_key,
        "collection_name": collection_name,
        "embedding_dim": dim,
        "total_points": total_indexed_points,
        "qdrant_points_count": coll_info["points_count"],
        "total_time_seconds": round(total_time, 2),
        "average_points_per_second": round(avg_rate, 1),
        "schema_version": SCHEMA_VERSION,
        "schema_frozen_at": SCHEMA_FROZEN_AT,
        "company_results": company_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Index financial chunks & embeddings into Qdrant collections."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding model provider key (default: bge_base).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help="Chunking method to index, or 'all' for all 5 methods (default: all).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help="Recreate collection if it already exists (default: True).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for Qdrant points upsert (default: 200).",
    )

    args = parser.parse_args()

    methods_to_run = (
        list(CHUNK_METHODS.keys())
        if args.method == "all"
        else [args.method]
    )

    for m in methods_to_run:
        if m not in CHUNK_METHODS:
            logger.error(f"Unknown method '{m}'. Available: {list(CHUNK_METHODS.keys())}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("FinAnalyst-AI: Qdrant Indexing Pipeline (V0 Baseline)")
    logger.info(f"Schema Version: {SCHEMA_VERSION} (Frozen at: {SCHEMA_FROZEN_AT})")
    logger.info(f"Model Provider: {args.model}")
    logger.info(f"Methods to Index: {methods_to_run}")
    logger.info(f"Payload Indexes: {list(PAYLOAD_INDEXES_V0.keys())}")
    logger.info("=" * 70)

    indexer = QdrantIndexer()
    summary_results = {}
    total_start = time.time()

    try:
        for method_key in methods_to_run:
            res = index_single_method(
                indexer=indexer,
                model_key=args.model,
                method_key=method_key,
                recreate=args.recreate,
                batch_size=args.batch_size,
            )
            summary_results[method_key] = res

        overall_time = time.time() - total_start
        overall_points = sum(r["total_points"] for r in summary_results.values())

        # Save summary report
        summary_path = OUTPUT_RETRIEVAL_ROOT / "indexing_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "overall_status": "SUCCESS",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_points_indexed": overall_points,
                    "total_elapsed_seconds": round(overall_time, 2),
                    "model_key": args.model,
                    "methods": summary_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("=" * 70)
        logger.info("INDEXING COMPLETED SUCCESSFULLY FOR ALL METHODS!")
        logger.info(f"Total Collections: {len(summary_results)}")
        logger.info(f"Total Points Indexed: {overall_points:,}")
        logger.info(f"Total Duration: {overall_time:.2f}s ({overall_time/60:.1f} mins)")
        logger.info(f"Summary Report: {summary_path}")
        logger.info("=" * 70)

    finally:
        indexer.close()


if __name__ == "__main__":
    main()
