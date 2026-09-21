"""Configuration for FinAnalyst-AI RAG Pipeline (V0 Baseline).

Supports multi-model embedding via EMBEDDING_PROVIDERS registry.
"""

from pathlib import Path
from typing import Dict, Any, List

# --- Project Paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
FINANALYST_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = FINANALYST_ROOT / "data"
GOLD_TEST_SET_DIR = DATA_DIR / "gold_test_set"
GOLD_TEST_SET_FILE = GOLD_TEST_SET_DIR / "v0_gold_questions.jsonl"

OUTPUT_CHUNKING_ROOT = FINANALYST_ROOT / "output_chunking"
OUTPUT_RETRIEVAL_ROOT = FINANALYST_ROOT / "output_retrieval" / "v0_dense_baseline"
EMBEDDINGS_CACHE_DIR = OUTPUT_RETRIEVAL_ROOT / "embeddings_cache"

# --- Embedding Provider Registry ---
# Each provider defines a complete set of model-specific parameters.
# Add new models here to benchmark them against existing ones.
EMBEDDING_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "qwen3": {
        "display_name": "Qwen3-Embedding-0.6B",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_dim": 1024,
        "max_seq_length": 1024,  # Optimized for 4GB VRAM (covers 93%+ of chunks)
        "batch_size": 4,        # Safe batch size for RTX 3050 Ti Laptop (4GB VRAM)
    },
    "bge_m3": {
        "display_name": "BGE-M3 (BAAI)",
        "model_name": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "max_seq_length": 1024,
        "batch_size": 4,
    },
}

DEFAULT_EMBEDDING_PROVIDER = "qwen3"


def get_provider_config(model_key: str) -> Dict[str, Any]:
    """Get the full configuration dict for a given model key."""
    if model_key not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider '{model_key}'. "
            f"Available: {list(EMBEDDING_PROVIDERS.keys())}"
        )
    return EMBEDDING_PROVIDERS[model_key]


def get_embeddings_cache_dir(model_key: str) -> Path:
    """Return the model-scoped cache directory for embedding vectors."""
    return EMBEDDINGS_CACHE_DIR / model_key


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

# --- Companies List ---
COMPANIES: List[str] = [
    "amd_10k_2025",
    "apple_2025_10k",
    "intel_2025_10k",
    "nvidia_2025_10k",
]

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
    "method2_5_heading_llm": {
        "display_name": "Method 2.5: Heading + LLM",
        "folder_name": "method2_5_heading_llm",
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

