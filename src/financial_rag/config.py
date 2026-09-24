"""Configuration for FinAnalyst-AI RAG Pipeline (V0 Baseline).

Supports multi-model embedding via EMBEDDING_PROVIDERS registry.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

from finanalyst.paths import (
    DATA_DIR,
    GOLD_TEST_SET_DIR,
    GOLD_TEST_SET_FILE,
    OUTPUT_CHUNKING,
    OUTPUT_RETRIEVAL,
    PROJECT_ROOT,
)

# --- Project Paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
FINANALYST_ROOT = PROJECT_ROOT
OUTPUT_CHUNKING_ROOT = OUTPUT_CHUNKING
OUTPUT_RETRIEVAL_ROOT = OUTPUT_RETRIEVAL / "v0_dense_baseline"
OUTPUT_RETRIEVAL_V1 = OUTPUT_RETRIEVAL / "v1_hybrid"
DEFAULT_RETRIEVAL_MODE = "hybrid"
SPARSE_MODEL_NAME = "Qdrant/bm25"
EMBEDDINGS_CACHE_DIR = OUTPUT_RETRIEVAL_ROOT / "embeddings_cache"

# --- Embedding Provider Registry ---
# Each provider defines a complete set of model-specific parameters.
# Add new models here to benchmark them against existing ones.
EMBEDDING_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "bge_base": {
        "display_name": "BGE-Base-EN-v1.5 (BAAI)",
        "model_name": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "max_seq_length": 512,
        "batch_size": 32,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "qwen3": {
        "display_name": "Qwen3-Embedding-0.6B",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "max_seq_length": 1024,  # Optimized for 4GB VRAM
        "batch_size": 8,
        "query_prefix": "",
    },
    "bge_m3": {
        "display_name": "BGE-M3 (BAAI)",
        "model_name": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "max_seq_length": 2048,
        "batch_size": 8,
        "query_prefix": "",
    },
}

DEFAULT_EMBEDDING_PROVIDER = "bge_base"


def get_provider_config(model_key: str) -> Dict[str, Any]:
    """Get the full configuration dict for a given model key."""
    if model_key not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider '{model_key}'. "
            f"Available: {list(EMBEDDING_PROVIDERS.keys())}"
        )
    return EMBEDDING_PROVIDERS[model_key]


def get_embeddings_cache_dir(model_key: str, method_key: Optional[str] = None) -> Path:
    """Return the model-scoped cache directory for embedding vectors."""
    base = EMBEDDINGS_CACHE_DIR / model_key
    if method_key:
        return base / method_key
    return base


def get_collection_name(model_key: str, method_key: str) -> str:
    """Generate a Qdrant collection name scoped by model and method.

    Examples:
        get_collection_name("qwen3", "method1_fixed_size")
        -> "finanalyst_v0_qwen3_method1_fixed_size"
    """
    return f"finanalyst_v0_{model_key}_{method_key}"


# --- Backward Compatibility Constants ---
# Used by any code that hasn't been refactored to accept model_key yet.
_default_provider = EMBEDDING_PROVIDERS[DEFAULT_EMBEDDING_PROVIDER]
EMBEDDING_MODEL_NAME = _default_provider["model_name"]
EMBEDDING_DIM = _default_provider["embedding_dim"]
MAX_SEQ_LENGTH = _default_provider["max_seq_length"]
BATCH_SIZE = _default_provider["batch_size"]

# --- Qdrant Vector Database Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_STORAGE_PATH = DATA_DIR / "qdrant_db"
QDRANT_PREFER_LOCAL = True

from financial_chunker.config import get_company_specs

# --- Companies List ---
COMPANIES: List[str] = [spec["dir_name"] for spec in get_company_specs(language="en")]

# --- 5 Chunking Methods Mapping ---
CHUNK_METHODS: Dict[str, Dict[str, Any]] = {
    "method1_fixed_size": {
        "display_name": "Method 1: Fixed-Size (512 tokens)",
        "folder_name": "baseline_fixed_size",
        "retrieval_text_key": "content",
    },
    "method2_deterministic": {
        "display_name": "Method 2: Deterministic Layout",
        "folder_name": "baseline_deterministic_structure",
        "retrieval_text_key": "content",
    },
    "method3_heading_llm": {
        "display_name": "Method 3: Heading + LLM",
        "folder_name": "method3_heading_llm",
        "retrieval_text_key": "content",
    },
    "method4_boundary_tagging": {
        "display_name": "Method 4: Boundary Tagging",
        "folder_name": "method4_boundary_tagging",
        "retrieval_text_key": "content",
    },
    "method5_proposed_golden_hybrid": {
        "display_name": "Method 5: Proposed Golden Hybrid",
        "folder_name": "method5_proposed_golden_hybrid",
        "retrieval_text_key": "content_retrieval",
    },
}

# --- Evaluation Metrics Constants ---
RETRIEVAL_TOP_K = 50
EVAL_CUTOFFS = [5, 10, 20, 50]

# --- Retrieval Prefetch & Filter Strategy ---
FILTER_STRATEGY = "post_filter"  # Post-filter applied symmetrically to both Dense and BM25
DENSE_PREFETCH_DEPTH = 200
BM25_PREFETCH_DEPTH = 200
PREFETCH_RETRY_DEPTH = 400
MAX_RETRY_ON_EMPTY_FILTER = 1

# --- RRF Fusion Configuration ---
RRF_K = 60
SEARCH_MODES = ["dense", "bm25", "hybrid"]

# --- BM25 Parameters & Cache Versioning ---
BM25_K1 = 1.5
BM25_B = 0.75
BM25_CACHE_DIR = OUTPUT_RETRIEVAL_ROOT / "bm25_cache"
BM25_CACHE_VERSION = "v1_keep_numbers_no_stem"
TOKENIZER_VERSION = "financial_v1"
BM25_USE_STEMMING = False

# --- Rescue Metrics Configuration ---
RESCUE_METRICS_ENABLED = True
RESCUE_TOP_K = 10

# --- Financial Stopwords ---
FINANCIAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "do", "does", "did", "this", 
    "that", "these", "those", "it", "its", "we", "our", "which", "what"
}

