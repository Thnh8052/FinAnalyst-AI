"""Master End-to-End Runner for FinAnalyst-AI Financial RAG Pipeline (V1 Native Hybrid & V0 Dense).

Supports multi-model embedding benchmark via --model argument and hybrid/dense search modes via --mode.

Usage:
    # Run entire pipeline with V1 Hybrid retrieval (BGE-base + Qdrant BM25 RRF):
    python src/financial_rag/run_pipeline.py --stage all --model bge_base --mode hybrid

    # Run sparse lexical indexing for existing Qdrant collections:
    python src/financial_rag/run_pipeline.py --stage index_sparse --model bge_base

    # Run retrieval evaluation:
    python src/financial_rag/run_pipeline.py --stage evaluate --model bge_base --mode hybrid
    python src/financial_rag/run_pipeline.py --stage evaluate --model bge_base --mode dense
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure src/ is on sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parents[2]
_SRC = ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from financial_rag.config import (
    CHUNK_METHODS,
    COMPANIES,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_RETRIEVAL,
    OUTPUT_RETRIEVAL_ROOT,
    GOLD_TEST_SET_FILE,
    EMBEDDING_PROVIDERS,
    DEFAULT_EMBEDDING_PROVIDER,
    get_provider_config,
    get_embeddings_cache_dir,
    get_collection_name,
)
from financial_rag.embeddings import (
    EmbeddingEngine,
    extract_retrieval_text,
    save_cached_embeddings,
    load_cached_embeddings,
)
from financial_rag.indexing import QdrantIndexer
from financial_rag.retrieval import HybridRetriever, DenseRetriever
from financial_rag.testbed import load_gold_test_set
from financial_rag.evaluation import RetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAGPipeline")


def load_chunks_for_method(method_key: str) -> List[Dict[str, Any]]:
    """Load all chunks for a given method across all 35 companies/filings."""
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


def run_embedding_stage(methods: List[str], model_key: str) -> None:
    """Stage 1: Generate and cache dense vector embeddings for all methods."""
    provider = get_provider_config(model_key)
    cache_dir = get_embeddings_cache_dir(model_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== STAGE 1: GENERATING VECTOR EMBEDDINGS [{provider['display_name']}] ===")
    embedder = EmbeddingEngine.get_instance(model_key)

    for method_key in methods:
        method_info = CHUNK_METHODS[method_key]
        cache_file = cache_dir / f"{method_key}_embeddings.npz"

        if cache_file.exists():
            logger.info(f"Embeddings cache already exists for {method_key}: {cache_file}")
            continue

        chunks = load_chunks_for_method(method_key)
        if not chunks:
            continue

        text_key = method_info["retrieval_text_key"]
        texts = [extract_retrieval_text(c, text_key=text_key) for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks for {method_info['display_name']} (text_key='{text_key}')...")
        start_t = time.perf_counter()
        vectors = embedder.encode_texts(texts, batch_size=provider["batch_size"], show_progress=True)
        elapsed = time.perf_counter() - start_t
        logger.info(f"Encoded {len(texts)} chunks in {elapsed:.2f}s ({len(texts)/elapsed:.1f} chunks/sec).")

        save_cached_embeddings(cache_file, chunk_ids, vectors)


def run_indexing_stage(methods: List[str], model_key: str, recreate: bool = False) -> None:
    """Stage 2: Index chunks and dense embeddings into Qdrant collections."""
    provider = get_provider_config(model_key)
    cache_dir = get_embeddings_cache_dir(model_key)

    logger.info(f"=== STAGE 2: INDEXING INTO QDRANT VECTOR DB [{provider['display_name']}] ===")
    indexer = QdrantIndexer()

    for method_key in methods:
        coll_name = get_collection_name(model_key, method_key)
        cache_file = cache_dir / f"{method_key}_embeddings.npz"

        chunks = load_chunks_for_method(method_key)
        if not chunks:
            continue

        cached = load_cached_embeddings(cache_file)
        if not cached:
            logger.error(f"Embeddings not found in cache for {method_key}. Run --stage embed first.")
            continue

        chunk_ids, vectors = cached
        indexer.init_collection(coll_name, recreate=recreate)
        indexed_count = index_chunks_with_mapping(indexer, coll_name, chunks, vectors, method_key)

        stats = indexer.get_collection_info(coll_name)
        logger.info(f"Collection '{coll_name}' stats: {stats}")


def run_sparse_indexing_stage(methods: List[str], model_key: str, batch_size: int = 256) -> None:
    """Stage 2.5: Populate named 'sparse' vectors using FastEmbed Qdrant/bm25."""
    logger.info("=== STAGE 2.5: SPARSE LEXICAL INDEXING (Qdrant/bm25) ===")
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        logger.error("fastembed is not installed. Please run: pip install fastembed")
        return

    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    indexer = QdrantIndexer()

    for method_key in methods:
        method_info = CHUNK_METHODS[method_key]
        coll_name = get_collection_name(model_key, method_key)
        disp_name = method_info["display_name"]
        text_key = method_info["retrieval_text_key"]

        logger.info(f"Generating FastEmbed BM25 vectors for {disp_name} ({coll_name})...")
        chunks = load_chunks_for_method(method_key)
        if not chunks:
            continue

        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [extract_retrieval_text(c, text_key=text_key) for c in chunks]
        total_chunks = len(texts)

        start_t = time.perf_counter()
        sparse_vectors = []
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embs = list(sparse_model.embed(batch_texts))
            sparse_vectors.extend(batch_embs)

        emb_time = time.perf_counter() - start_t
        logger.info(f"Generated {total_chunks} sparse embeddings in {emb_time:.2f}s ({total_chunks/emb_time:.1f} vec/s).")

        logger.info(f"Updating sparse vectors in Qdrant collection '{coll_name}'...")
        updated = indexer.update_sparse_vectors(coll_name, chunk_ids, sparse_vectors)
        logger.info(f"Updated {updated} points in '{coll_name}'.")

    indexer.close()


def index_chunks_with_mapping(
    indexer: QdrantIndexer,
    coll_name: str,
    chunks: List[Dict[str, Any]],
    vectors,
    method_key: str,
) -> int:
    """Index chunks matching chunk_id order."""
    return indexer.index_chunks(
        collection_name=coll_name,
        chunks=chunks,
        vectors=vectors,
        method_key=method_key,
        batch_size=100,
    )


def run_evaluation_stage(methods: List[str], model_key: str, mode: str = "hybrid") -> None:
    """Stage 3: Run retrieval benchmark against Gold Test Set."""
    provider = get_provider_config(model_key)

    logger.info(f"=== STAGE 3: RUNNING RETRIEVAL BENCHMARK [{mode.upper()} | {provider['display_name']}] ===")
    questions = load_gold_test_set(GOLD_TEST_SET_FILE)
    if not questions:
        logger.error(f"No gold questions found at {GOLD_TEST_SET_FILE}. Please populate test set.")
        return

    evaluator = RetrievalEvaluator(retrieval_mode=mode)
    out_dir = OUTPUT_RETRIEVAL / ("v1_hybrid" if mode == "hybrid" else f"v0_{mode}_baseline")
    results = evaluator.run_benchmark(
        gold_questions=questions,
        methods=methods,
        model_key=model_key,
        output_dir=out_dir,
    )

    logger.info("=== BENCHMARK COMPLETE ===")
    report_file = "V1_HYBRID_RETRIEVAL_REPORT.md" if mode == "hybrid" else "V0_RETRIEVAL_EVALUATION_REPORT.md"
    logger.info(f"Master report: {out_dir / report_file}")


def main():
    parser = argparse.ArgumentParser(description="FinAnalyst-AI RAG Pipeline Master Runner (V1 Hybrid)")
    parser.add_argument(
        "--stage",
        choices=["all", "embed", "index", "index_sparse", "evaluate"],
        default="all",
        help="Stage of the pipeline to execute.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated list of method keys to process, or 'all'.",
    )
    parser.add_argument(
        "--model",
        choices=list(EMBEDDING_PROVIDERS.keys()),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help=f"Embedding model provider to use (default: '{DEFAULT_EMBEDDING_PROVIDER}').",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Retrieval search mode for evaluation (default: 'hybrid').",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections if they already exist.",
    )

    args = parser.parse_args()

    target_methods = list(CHUNK_METHODS.keys()) if args.methods == "all" else [
        m.strip() for m in args.methods.split(",") if m.strip() in CHUNK_METHODS
    ]
    model_key = args.model
    mode = args.mode

    provider = get_provider_config(model_key)
    logger.info(f"Pipeline config: model={provider['display_name']}, mode={mode}, methods={target_methods}")

    if args.stage in ["all", "embed"]:
        run_embedding_stage(target_methods, model_key)

    if args.stage in ["all", "index"]:
        run_indexing_stage(target_methods, model_key, recreate=args.recreate)

    if args.stage in ["all", "index_sparse"]:
        run_sparse_indexing_stage(target_methods, model_key)

    if args.stage in ["all", "evaluate"]:
        run_evaluation_stage(target_methods, model_key, mode=mode)


if __name__ == "__main__":
    main()
