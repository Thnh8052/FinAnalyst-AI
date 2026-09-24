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

from financial_rag.config import (
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

        self.query_prefix = provider.get("query_prefix", "")

        import os
        env_dev = os.environ.get("EMBEDDING_DEVICE", "").strip().lower()
        if env_dev == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading embedding model '{self.model_name}' [{model_key}] "
            f"target device='{self.device}'..."
        )
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs={"use_safetensors": False},
            )
            if self.device == "cuda":
                try:
                    self.model = self.model.to(torch.device("cuda"))
                    # Quick forward pass test to verify VRAM headroom
                    _ = self.model.encode("warmup", show_progress_bar=False)
                except Exception as cuda_err:
                    logger.warning(f"CUDA memory allocation failed ({cuda_err}), falling back to CPU...")
                    self.model = self.model.to(torch.device("cpu"))
                    self.device = "cpu"
        except Exception as e:
            logger.warning(f"Staged load failed ({e}), falling back to direct CPU load...")
            self.model = SentenceTransformer(
                self.model_name,
                device="cpu",
                model_kwargs={"use_safetensors": False},
            )
            self.device = "cpu"
        self.model.max_seq_length = self.max_seq_length
        logger.info(
            f"Model loaded successfully. Dim={self.embedding_dim}, "
            f"MaxSeqLen={self.max_seq_length}, BatchSize={self.batch_size}, "
            f"QueryPrefix='{self.query_prefix}'"
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
        """Encode a single retrieval query into a 1D vector with instruction prefix."""
        prefixed_query = query
        if self.query_prefix and not query.startswith(self.query_prefix):
            prefixed_query = self.query_prefix + query

        emb = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb

    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode multiple retrieval queries into a 2D matrix with instruction prefix."""
        prefixed_queries = [
            self.query_prefix + q if self.query_prefix and not q.startswith(self.query_prefix) else q
            for q in queries
        ]
        return self.encode_texts(prefixed_queries, batch_size=batch_size, show_progress=False)


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
    """Save embeddings and corresponding chunk IDs to compressed numpy archive (fp16)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        chunk_ids=np.array(chunk_ids, dtype=object),
        embeddings=embeddings.astype(np.float16),
    )
    logger.info(f"Saved {len(chunk_ids)} cached embeddings (fp16) to {cache_path}")


def load_cached_embeddings(cache_path: Path) -> Optional[Tuple[List[str], np.ndarray]]:
    """Load cached embeddings and chunk IDs from file if exists."""
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    chunk_ids = list(data["chunk_ids"])
    embeddings = data["embeddings"].astype(np.float32)
    logger.info(f"Loaded {len(chunk_ids)} cached embeddings from {cache_path}")
    return chunk_ids, embeddings

