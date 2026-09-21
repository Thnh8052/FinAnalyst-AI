"""Master End-to-End Runner for FinAnalyst-AI Financial RAG Pipeline.

Supports multi-model embedding benchmark via --model argument.

Usage:
    # Run entire pipeline from embedding to evaluation:
    python script/financial_rag/run_pipeline.py --stage all --model qwen3

    # Run specific stages with specific model:
    python script/financial_rag/run_pipeline.py --stage embed --model qwen3
    python script/financial_rag/run_pipeline.py --stage embed --model bge_m3
    python script/financial_rag/run_pipeline.py --stage index --model qwen3 --recreate
    python script/financial_rag/run_pipeline.py --stage evaluate --model qwen3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from script.financial_rag.config import (
    CHUNK_METHODS,
    COMPANIES,
    OUTPUT_CHUNKING_ROOT,
    OUTPUT_RETRIEVAL_ROOT,
    GOLD_TEST_SET_FILE,
    EMBEDDING_PROVIDERS,
    DEFAULT_EMBEDDING_PROVIDER,
    get_provider_config,
    get_embeddings_cache_dir,
    get_collection_name,
)
from script.financial_rag.embeddings import (
    EmbeddingEngine,
    extract_retrieval_text,
    save_cached_embeddings,
    load_cached_embeddings,
)
from script.financial_rag.indexing import QdrantIndexer
from script.financial_rag.retrieval import DenseRetriever
from script.financial_rag.testbed import load_gold_test_set
from script.financial_rag.evaluation import RetrievalEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAGPipeline")


def load_chunks_for_method(method_key: str) -> List[Dict[str, Any]]:
    """Load all chunks for a given method across all 4 companies."""
    folder_name = CHUNK_METHODS[method_key]["folder_name"]
    all_chunks: List[Dict[str, Any]] = []

    for comp in COMPANIES:
        chunk_file = OUTPUT_CHUNKING_ROOT / comp / folder_name / "chunks.jsonl"
        if not chunk_file.exists():
            logger.warning(f"File not found: {chunk_file}")
            continue

        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_chunks.append(json.loads(line))

    logger.info(f"Loaded {len(all_chunks)} chunks for {method_key} across {len(COMPANIES)} companies.")
    return all_chunks


def run_embedding_stage(methods: List[str], model_key: str) -> None:
    """Stage 1: Generate and cache vector embeddings for all methods.

    Embeddings are saved to a model-scoped subdirectory:
        embeddings_cache/{model_key}/{method_key}_embeddings.npz
    """
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
    """Stage 2: Index chunks and embeddings into Qdrant collections.

    Collection names are scoped by model:
        finanalyst_v0_{model_key}_{method_key}
    """
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

        # Load vectors from cache
        cached = load_cached_embeddings(cache_file)
        if not cached:
            logger.error(f"Embeddings not found in cache for {method_key}. Run --stage embed first.")
            continue

        chunk_ids, vectors = cached
        indexer.init_collection(coll_name, recreate=recreate)
        indexed_count = index_chunks_with_mapping(indexer, coll_name, chunks, vectors, method_key)

        stats = indexer.get_collection_info(coll_name)
        logger.info(f"Collection '{coll_name}' stats: {stats}")


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


def run_evaluation_stage(methods: List[str], model_key: str) -> None:
    """Stage 3: Run retrieval benchmark against Gold Test Set."""
    provider = get_provider_config(model_key)

    logger.info(f"=== STAGE 3: RUNNING RETRIEVAL BENCHMARK [{provider['display_name']}] ===")
    questions = load_gold_test_set(GOLD_TEST_SET_FILE)
    if not questions:
        logger.error(f"No gold questions found at {GOLD_TEST_SET_FILE}. Please populate test set.")
        return

    evaluator = RetrievalEvaluator()
    results = evaluator.run_benchmark(
        gold_questions=questions,
        methods=methods,
        model_key=model_key,
        output_dir=OUTPUT_RETRIEVAL_ROOT,
    )

    logger.info("=== BENCHMARK COMPLETE ===")
    logger.info(f"Master report: {OUTPUT_RETRIEVAL_ROOT / 'V0_RETRIEVAL_EVALUATION_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description="FinAnalyst-AI RAG Pipeline Master Runner")
    parser.add_argument(
        "--stage",
        choices=["all", "embed", "index", "evaluate"],
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
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections if they already exist.",
    )

    args = parser.parse_args()

    target_methods = list(CHUNK_METHODS.keys()) if args.methods == "all" else [
        m.strip() for m in args.methods.split(",") if m.strip() in CHUNK_METHODS
    ]
    model_key = args.model

    provider = get_provider_config(model_key)
    logger.info(f"Pipeline config: model={provider['display_name']}, methods={target_methods}")

    if args.stage in ["all", "embed"]:
        run_embedding_stage(target_methods, model_key)

    if args.stage in ["all", "index"]:
        run_indexing_stage(target_methods, model_key, recreate=args.recreate)

    if args.stage in ["all", "evaluate"]:
        run_evaluation_stage(target_methods, model_key)


if __name__ == "__main__":
    main()

