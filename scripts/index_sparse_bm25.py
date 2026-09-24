"""Fast Sparse Lexical Indexer (FastEmbed Qdrant/bm25 -> Qdrant Vector DB).

Populates the named 'sparse' vector index for existing Qdrant collections
WITHOUT re-embedding or modifying existing dense vectors.

Usage:
    # Index all 5 chunking methods:
    python scripts/index_sparse_bm25.py --methods all --model bge_base

    # Index only the proposed Method 5:
    python scripts/index_sparse_bm25.py --methods method5_proposed_golden_hybrid --model bge_base
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC = ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastembed import SparseTextEmbedding

from financial_rag.config import (
    CHUNK_METHODS,
    COMPANIES,
    DEFAULT_EMBEDDING_PROVIDER,
    OUTPUT_CHUNKING_ROOT,
    get_collection_name,
)
from financial_rag.embeddings import extract_retrieval_text
from financial_rag.indexing import QdrantIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SparseIndexer")


def load_chunks_for_method(method_key: str) -> List[Dict[str, Any]]:
    """Load all chunks for a given method across all 35 company filings."""
    folder_name = CHUNK_METHODS[method_key]["folder_name"]
    all_chunks: List[Dict[str, Any]] = []

    for comp in COMPANIES:
        chunk_file = OUTPUT_CHUNKING_ROOT / comp / folder_name / "chunks.jsonl"
        if not chunk_file.exists():
            continue

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    logger.info(f"Loaded {len(all_chunks)} chunks for {method_key} across company filings.")
    return all_chunks


def run_sparse_indexing(
    target_methods: List[str],
    model_key: str = DEFAULT_EMBEDDING_PROVIDER,
    batch_size: int = 256,
) -> None:
    logger.info("Initializing FastEmbed SparseTextEmbedding(model_name='Qdrant/bm25')...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    indexer = QdrantIndexer()
    logger.info(f"Connected to Qdrant (local={indexer.is_local})")

    for method_key in target_methods:
        if method_key not in CHUNK_METHODS:
            logger.warning(f"Unknown method '{method_key}'. Skipping.")
            continue

        method_info = CHUNK_METHODS[method_key]
        coll_name = get_collection_name(model_key, method_key)
        disp_name = method_info["display_name"]
        text_key = method_info["retrieval_text_key"]

        logger.info(f"\n{'='*70}\n[SPARSE INDEXING] {disp_name}\nTarget Collection: {coll_name}\n{'='*70}")

        chunks = load_chunks_for_method(method_key)
        if not chunks:
            logger.warning(f"No chunks found for {method_key}. Skipping.")
            continue

        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [extract_retrieval_text(c, text_key=text_key) for c in chunks]
        total_chunks = len(texts)

        logger.info(f"Generating FastEmbed BM25 sparse vectors for {total_chunks} chunks...")
        start_t = time.perf_counter()

        # Batch compute sparse embeddings
        sparse_vectors = []
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embs = list(sparse_model.embed(batch_texts))
            sparse_vectors.extend(batch_embs)
            if (i + batch_size) % 2048 < batch_size or (i + batch_size) >= total_chunks:
                logger.info(f"  Processed {min(i + batch_size, total_chunks)}/{total_chunks} sparse embeddings...")

        emb_time = time.perf_counter() - start_t
        logger.info(
            f"Generated {total_chunks} sparse vectors in {emb_time:.2f}s "
            f"({total_chunks / emb_time:.1f} vectors/sec)."
        )

        # Update vectors in Qdrant collection
        logger.info(f"Updating named 'sparse' vectors in collection '{coll_name}'...")
        start_upsert = time.perf_counter()
        updated_count = indexer.update_sparse_vectors(
            collection_name=coll_name,
            chunk_ids=chunk_ids,
            sparse_vectors=sparse_vectors,
            batch_size=250,
        )
        upsert_time = time.perf_counter() - start_upsert
        logger.info(
            f"Successfully updated {updated_count} sparse vectors in {upsert_time:.2f}s "
            f"({updated_count / upsert_time:.1f} pts/sec)."
        )

        stats = indexer.get_collection_info(coll_name)
        logger.info(f"Collection status after update: {stats}")

    indexer.close()
    logger.info("\n[SUCCESS] Sparse lexical indexing complete for all target collections!")


def main():
    parser = argparse.ArgumentParser(description="Populate Qdrant Sparse Named Vectors with FastEmbed BM25")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated list of method keys, or 'all'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding model key (default: '{DEFAULT_EMBEDDING_PROVIDER}').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sparse embedding generation (default: 256).",
    )
    args = parser.parse_args()

    target_methods = list(CHUNK_METHODS.keys()) if args.methods == "all" else [
        m.strip() for m in args.methods.split(",") if m.strip() in CHUNK_METHODS
    ]

    run_sparse_indexing(target_methods, model_key=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
