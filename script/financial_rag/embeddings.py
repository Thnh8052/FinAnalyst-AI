"""Embedding Engine with multi-model support via EmbeddingProvider registry.

Supports switching between embedding models (e.g., Qwen3-Embedding-0.6B, BGE-M3)
through the EMBEDDING_PROVIDERS registry in config.py.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from script.financial_rag.config import (
    EMBEDDING_PROVIDERS,
    DEFAULT_EMBEDDING_PROVIDER,
    get_provider_config,
)

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages model loading and batch embedding generation for financial chunks.

    Uses a registry-based provider pattern: pass a model_key (e.g., "qwen3" or
    "bge_m3") to load the corresponding model from EMBEDDING_PROVIDERS.
    """

    _instances: Dict[str, "EmbeddingEngine"] = {}

    def __init__(self, model_key: str = DEFAULT_EMBEDDING_PROVIDER):
        provider = get_provider_config(model_key)

        self.model_key = model_key
        self.model_name = provider["model_name"]
        self.embedding_dim = provider["embedding_dim"]
        self.max_seq_length = provider["max_seq_length"]
        self.batch_size = provider["batch_size"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(
            f"Loading embedding model '{self.model_name}' [{model_key}] "
            f"on device='{self.device}' (dtype={dtype})..."
        )
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"torch_dtype": dtype},
            device=self.device,
        )
        self.model.max_seq_length = self.max_seq_length
        logger.info(
            f"Model loaded successfully. Dim={self.embedding_dim}, "
            f"MaxSeqLen={self.max_seq_length}, BatchSize={self.batch_size}"
        )

    @classmethod
    def get_instance(cls, model_key: str = DEFAULT_EMBEDDING_PROVIDER) -> "EmbeddingEngine":
        """Get or initialize a singleton instance keyed by model_key.

        On a 4GB GPU, only one model should be loaded at a time. If switching
        models, call release_instance() on the old one first.
        """
        if model_key not in cls._instances:
            cls._instances[model_key] = cls(model_key)
        return cls._instances[model_key]

    @classmethod
    def release_instance(cls, model_key: str) -> None:
        """Release a model instance to free GPU memory before loading another."""
        if model_key in cls._instances:
            del cls._instances[model_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Released embedding model instance '{model_key}'.")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of text strings into normalized vector embeddings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for GPU inference (defaults to provider config).
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        bs = batch_size or self.batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single retrieval query into a 1D vector."""
        emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb


def extract_retrieval_text(chunk: dict, text_key: str = "content") -> str:
    """Extract text to be embedded from a chunk dictionary.

    For Method 5, uses 'content_retrieval' (which contains structured semantic tuples
    and markdown table representations). Falls back to 'content' if not present.
    """
    text = chunk.get(text_key)
    if not text or not str(text).strip():
        text = chunk.get("content", "")
    return str(text).strip()


def save_cached_embeddings(
    cache_path: Path,
    chunk_ids: List[str],
    embeddings: np.ndarray,
) -> None:
    """Save embeddings and corresponding chunk IDs to compressed numpy archive."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        chunk_ids=np.array(chunk_ids, dtype=object),
        embeddings=embeddings,
    )
    logger.info(f"Saved {len(chunk_ids)} cached embeddings to {cache_path}")


def load_cached_embeddings(cache_path: Path) -> Optional[Tuple[List[str], np.ndarray]]:
    """Load cached embeddings and chunk IDs from file if exists."""
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    chunk_ids = list(data["chunk_ids"])
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(chunk_ids)} cached embeddings from {cache_path}")
    return chunk_ids, embeddings

