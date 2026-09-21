"""Qdrant Vector Database Indexer for Financial Chunks."""

import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType

from script.financial_rag.config import (
    QDRANT_URL,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


def generate_point_id(chunk_id: str) -> str:
    """Generate a deterministic UUID string from chunk_id for idempotent Qdrant points."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantIndexer:
    """Manages collection lifecycle and batch upserting into Qdrant."""

    def __init__(self, url: str = QDRANT_URL):
        self.url = url
        self.client = QdrantClient(url=self.url)

    def init_collection(self, collection_name: str, recreate: bool = False) -> None:
        """Create Qdrant collection with cosine distance and payload indexes."""
        existing = [c.name for c in self.client.get_collections().collections]

        if collection_name in existing:
            if recreate:
                logger.info(f"Recreating collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
                return

        logger.info(f"Creating collection '{collection_name}' (dim={EMBEDDING_DIM}, distance=Cosine)...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        # Create payload indexes for metadata filtering
        index_fields = [
            ("ticker", PayloadSchemaType.KEYWORD),
            ("fiscal_year", PayloadSchemaType.INTEGER),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("statement_type", PayloadSchemaType.KEYWORD),
            ("chunk_method", PayloadSchemaType.KEYWORD),
        ]

        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.warning(f"Could not create index for field '{field_name}': {e}")

        logger.info(f"Collection '{collection_name}' initialized with payload indexes.")

    @staticmethod
    def build_payload(chunk: Dict[str, Any], method_key: str) -> Dict[str, Any]:
        """Construct standard Qdrant payload from chunk dict."""
        meta = chunk.get("metadata", {}) or {}
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("document_id", ""),
            "ticker": chunk.get("ticker", ""),
            "fiscal_year": int(chunk.get("fiscal_year", 2025)),
            "chunk_method": method_key,
            "chunk_type": chunk.get("chunk_type", "unknown"),
            "statement_type": meta.get("statement_type", "other"),
            "section_title": meta.get("section_title", ""),
            "source_pages": chunk.get("source_pages", []),
            "token_count": int(chunk.get("token_count", 0)),
            "has_table_fragmentation": bool(chunk.get("has_table_fragmentation", False)),
            "content": chunk.get("content", ""),
            "content_retrieval": chunk.get("content_retrieval", ""),
            "content_generation": chunk.get("content_generation", ""),
        }

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
        method_key: str,
        batch_size: int = 100,
    ) -> int:
        """Batch upsert chunks and vectors into Qdrant.
        
        Returns:
            Number of points successfully indexed.
        """
        assert len(chunks) == len(vectors), f"Chunks count ({len(chunks)}) != Vectors count ({len(vectors)})"

        points: List[PointStruct] = []
        for i, chunk in enumerate(chunks):
            p_id = generate_point_id(chunk["chunk_id"])
            vector = vectors[i].tolist()
            payload = self.build_payload(chunk, method_key)
            points.append(PointStruct(id=p_id, vector=vector, payload=payload))

        total_points = len(points)
        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch = points[start_idx:end_idx]
            self.client.upsert(collection_name=collection_name, points=batch)

        logger.info(f"Indexed {total_points} chunks into collection '{collection_name}'.")
        return total_points

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics from Qdrant."""
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "status": str(info.status),
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        }
