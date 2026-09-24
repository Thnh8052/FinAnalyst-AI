"""Qdrant Vector Database Indexer for Financial Chunks.

Supports:
- Local Persistent Qdrant storage (default) or Server URL mode.
- Dual named vectors ('dense' 768-dim + 'sparse' on-disk index) for seamless V0 -> V1 transition.
- Standardized 7 payload indexes (ticker, fiscal_year, fiscal_period, statement_type, form_type, contains_table, contains_text).
- Dynamic computation of contains_table & contains_text via financial_chunker.enricher.compute_content_flags.
"""

import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    PayloadSchemaType,
)

from financial_chunker.enricher import compute_content_flags
from financial_rag.config import (
    QDRANT_URL,
    QDRANT_STORAGE_PATH,
    QDRANT_PREFER_LOCAL,
    EMBEDDING_DIM,
)
from financial_rag.schema import (
    SCHEMA_VERSION,
    SCHEMA_FROZEN_AT,
    CONTAINS_TABLE_MIN_TABLE_LINES,
    CONTAINS_TEXT_MIN_WORDS,
    EXCLUDED_PREFIXES,
    PAYLOAD_INDEXES_V0,
    TICKERS_V0,
    STATEMENT_TYPES,
    FISCAL_PERIODS,
    FORM_TYPES,
)

logger = logging.getLogger(__name__)

COMPANY_NAMES_MAP = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "NKE": "NIKE, Inc.",
    "WMT": "Walmart Inc.",
}


def generate_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID string from chunk_id for idempotent Qdrant points."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantIndexer:
    """Manages collection lifecycle, payload schemas, and batch upserting into Qdrant."""

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
        prefer_local: bool = QDRANT_PREFER_LOCAL,
    ):
        if prefer_local:
            target_path = Path(storage_path or QDRANT_STORAGE_PATH)
            target_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Connecting to Local Persistent Qdrant at: {target_path}")
            self.client = QdrantClient(path=str(target_path))
            self.is_local = True
        else:
            target_url = url or QDRANT_URL
            logger.info(f"Connecting to Server Qdrant at: {target_url}")
            self.client = QdrantClient(url=target_url)
            self.is_local = False

    def init_collection(
        self,
        collection_name: str,
        dim: int = EMBEDDING_DIM,
        recreate: bool = False,
    ) -> None:
        """Create Qdrant collection with dual named vectors ('dense' + 'sparse') and 7 payload indexes.
        
        Architecture guarantees:
        - Collection is born with both named vectors so V1 sparse vectors can be added via update_vectors
          without recreating the collection or re-indexing dense vectors.
        - Points in V0 only supply the 'dense' vector; Qdrant permits omitted sparse vectors gracefully.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if collection_name in existing:
            if recreate:
                logger.info(f"Recreating collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
                return

        logger.info(
            f"Creating collection '{collection_name}' with dual named vectors "
            f"(dense: dim={dim}, distance=Cosine, on_disk=True; sparse: on_disk=True)..."
        )
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=dim, distance=Distance.COSINE, on_disk=True)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
            },
        )

        # 9 Standardized Payload Indexes (Freeze V0 / Schema B)
        index_fields = [
            ("ticker", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("fiscal_year", PayloadSchemaType.INTEGER),
            ("period_years", PayloadSchemaType.INTEGER),  # Qdrant supports int array indexing
            ("fiscal_period", PayloadSchemaType.KEYWORD),
            ("statement_type", PayloadSchemaType.KEYWORD),
            ("form_type", PayloadSchemaType.KEYWORD),
            ("contains_table", PayloadSchemaType.BOOL),
            ("contains_text", PayloadSchemaType.BOOL),
        ]

        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Payload index creation note for '{field_name}': {e}")

        logger.info(f"Collection '{collection_name}' initialized with 9 payload indexes.")

    @staticmethod
    def build_payload(
        chunk: Dict[str, Any],
        method_key: str,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_version: str = "v1_hybrid",
    ) -> Dict[str, Any]:
        """Construct standardized Qdrant payload with content flags and financial metadata."""
        meta = chunk.get("metadata", {}) or {}
        content = chunk.get("content", "")
        chunk_type = chunk.get("chunk_type", "unknown")

        # Dynamically compute content flags: (contains_table, contains_text)
        has_table, has_text = compute_content_flags(
            chunk_text=content,
            chunk_type=chunk_type,
            min_table_lines=CONTAINS_TABLE_MIN_TABLE_LINES,
            narrative_word_threshold=CONTAINS_TEXT_MIN_WORDS,
            excluded_prefixes=EXCLUDED_PREFIXES,
        )

        # Extract unit and currency
        unit_meta = meta.get("unit", {})
        if isinstance(unit_meta, dict):
            currency = unit_meta.get("currency", "USD")
            unit_scale = unit_meta.get("scale", "million")
        else:
            currency = "USD"
            unit_scale = "million"

        statement_type = meta.get("statement_type", "other_financial")
        ticker = str(chunk.get("ticker") or meta.get("ticker") or "").upper()
        if not ticker and chunk.get("chunk_id"):
            cand = chunk["chunk_id"].split("_")[0].upper()
            if cand in COMPANY_NAMES_MAP or cand in ["AAPL", "NVDA", "AMZN", "AMD", "INTC", "NKE", "WMT"]:
                ticker = cand
        company_name = COMPANY_NAMES_MAP.get(ticker, ticker)

        doc_id = str(chunk.get("document_id") or meta.get("document_id") or "").lower()
        if not doc_id and chunk.get("chunk_id"):
            parts = str(chunk["chunk_id"]).lower().split("_")
            if len(parts) >= 2 and re.match(r"^20\d\d$", parts[1]):
                doc_id = f"{parts[0]}_{parts[1]}_10k"

        # FIX [B2]: Infer fiscal_year from doc_id instead of hardcoding 2025
        raw_fy = chunk.get("fiscal_year") or meta.get("fiscal_year")
        if raw_fy is not None and str(raw_fy).isdigit():
            fiscal_year = int(raw_fy)
        else:
            m_fy = re.search(r"\b(20\d\d)\b", doc_id)
            fiscal_year = int(m_fy.group(1)) if m_fy else 2025

        # FIX [B2]: Extract or infer period_years for multi-year financial statements
        period_years = meta.get("period_years") or chunk.get("period_years")
        if not period_years:
            found_years = {int(y) for y in re.findall(r"\b(20[12]\d)\b", content)}
            period_years = sorted(list(found_years)) if found_years else [fiscal_year]
        else:
            period_years = sorted(list({int(y) for y in period_years if str(y).isdigit()}))

        return {
            # --- Primary Identifiers ---
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": doc_id,
            "company": company_name,
            "chunk_method": method_key,
            "chunk_type": chunk_type,
            # --- Indexed Filter Fields (9 Fields) ---
            "ticker": ticker,
            "document_id": doc_id,
            "fiscal_year": fiscal_year,
            "period_years": period_years,
            "fiscal_period": str(chunk.get("fiscal_period", "FY")),
            "statement_type": statement_type,
            "form_type": str(chunk.get("form_type", "10-K")),
            "contains_table": has_table,
            "contains_text": has_text,
            # --- Extended Payload Fields (Display & Generation Provenance) ---
            "section_title": meta.get("section_title", ""),
            "source_pages": chunk.get("source_pages", []),
            "token_count": int(chunk.get("token_count", 0)),
            "currency": currency,
            "unit": unit_scale,
            "period_dates": meta.get("period_dates", []),
            "has_fact_tuples": bool(meta.get("has_linearized_tuples", False)),
            "has_table_fragmentation": bool(chunk.get("has_table_fragmentation", False)),
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "schema_version": SCHEMA_VERSION,
            # --- Content Bodies ---
            "content": content,
            "content_retrieval": chunk.get("content_retrieval", content),
            "content_generation": chunk.get("content_generation", content),
        }

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
        method_key: str,
        sparse_vectors: Optional[List[Any]] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_version: str = "v1_hybrid",
        batch_size: int = 100,
    ) -> int:
        """Batch upsert chunks and vectors (dense + optional sparse) into Qdrant.
        
        Args:
            collection_name: Name of target collection.
            chunks: List of chunk dictionaries loaded from chunks.jsonl.
            vectors: np.ndarray of shape (N, dim) dense embeddings.
            method_key: Chunking method identifier.
            sparse_vectors: Optional list of FastEmbed SparseEmbedding objects.
            embedding_model: Identifier of the embedding model used.
            embedding_version: Version identifier (e.g. 'v1_hybrid').
            batch_size: Number of points to upsert per batch.

        Returns:
            Number of points successfully indexed.
        """
        assert len(chunks) == len(vectors), (
            f"Chunks count ({len(chunks)}) != Vectors count ({len(vectors)})"
        )

        points: List[PointStruct] = []
        for i, chunk in enumerate(chunks):
            p_id = generate_point_id(chunk["chunk_id"])
            dense_vector = vectors[i].tolist()
            payload = self.build_payload(
                chunk,
                method_key=method_key,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )

            vector_dict: Dict[str, Any] = {"dense": dense_vector}
            if sparse_vectors is not None and i < len(sparse_vectors):
                s_vec = sparse_vectors[i]
                if hasattr(s_vec, "indices") and hasattr(s_vec, "values"):
                    idx = s_vec.indices.tolist() if hasattr(s_vec.indices, "tolist") else list(s_vec.indices)
                    val = s_vec.values.tolist() if hasattr(s_vec.values, "tolist") else list(s_vec.values)
                elif isinstance(s_vec, dict):
                    idx, val = s_vec["indices"], s_vec["values"]
                else:
                    idx, val = list(s_vec[0]), list(s_vec[1])
                vector_dict["sparse"] = rest.SparseVector(
                    indices=[int(x) for x in idx],
                    values=[float(x) for x in val],
                )

            points.append(
                PointStruct(
                    id=p_id,
                    vector=vector_dict,
                    payload=payload,
                )
            )

        total_points = len(points)
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch = points[start_idx:end_idx]
            self.client.upsert(collection_name=collection_name, points=batch)

        logger.info(f"Indexed {total_points} chunks into collection '{collection_name}'.")
        return total_points

    def update_sparse_vectors(
        self,
        collection_name: str,
        chunk_ids: List[str],
        sparse_vectors: List[Any],
        batch_size: int = 250,
    ) -> int:
        """Update or add named 'sparse' vectors for existing points in Qdrant collection.
        
        Args:
            collection_name: Name of target Qdrant collection.
            chunk_ids: List of chunk_ids corresponding to existing points.
            sparse_vectors: List of FastEmbed SparseEmbedding objects or dicts with indices & values.
            batch_size: Batch size for update_vectors calls.

        Returns:
            Number of points updated.
        """
        assert len(chunk_ids) == len(sparse_vectors), (
            f"chunk_ids count ({len(chunk_ids)}) != sparse_vectors count ({len(sparse_vectors)})"
        )

        points_to_update: List[rest.PointVectors] = []
        for cid, s_vec in zip(chunk_ids, sparse_vectors):
            p_id = generate_point_id(cid)
            if hasattr(s_vec, "indices") and hasattr(s_vec, "values"):
                indices = s_vec.indices.tolist() if hasattr(s_vec.indices, "tolist") else list(s_vec.indices)
                values = s_vec.values.tolist() if hasattr(s_vec.values, "tolist") else list(s_vec.values)
            elif isinstance(s_vec, dict):
                indices = s_vec["indices"]
                values = s_vec["values"]
            else:
                indices = list(s_vec[0])
                values = list(s_vec[1])

            points_to_update.append(
                rest.PointVectors(
                    id=p_id,
                    vector={
                        "sparse": rest.SparseVector(
                            indices=[int(idx) for idx in indices],
                            values=[float(val) for val in values],
                        )
                    },
                )
            )

        total_points = len(points_to_update)
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch = points_to_update[start_idx:end_idx]
            self.client.update_vectors(collection_name=collection_name, points=batch)

        logger.info(f"Updated sparse vectors for {total_points} chunks in '{collection_name}'.")
        return total_points

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics from Qdrant."""
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "status": str(info.status),
            "points_count": info.points_count,
            "vectors_count": getattr(info, "indexed_vectors_count", info.points_count),
        }

    def close(self) -> None:
        """Close client connection if applicable."""
        if hasattr(self.client, "close"):
            self.client.close()


# Module-level convenience aliases
build_payload = QdrantIndexer.build_payload
IndexManager = QdrantIndexer
