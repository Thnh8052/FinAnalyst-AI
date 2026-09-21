"""Dense Retrieval Engine for Qdrant Vector Collections."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from script.financial_rag.config import (
    QDRANT_URL,
    RETRIEVAL_TOP_K,
    DEFAULT_EMBEDDING_PROVIDER,
)
from script.financial_rag.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual chunk search hit."""
    rank: int
    score: float
    chunk_id: str
    ticker: str
    fiscal_year: int
    chunk_type: str
    statement_type: str
    section_title: str
    source_pages: List[int]
    content_snippet: str
    payload: Dict[str, Any]


class DenseRetriever:
    """Performs cosine dense vector search against Qdrant collections."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        model_key: str = DEFAULT_EMBEDDING_PROVIDER,
        embedder: Optional[EmbeddingEngine] = None,
    ):
        self.client = QdrantClient(url=url)
        self.model_key = model_key
        self.embedder = embedder or EmbeddingEngine.get_instance(model_key)

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        ticker: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform dense semantic search for a single natural language query."""
        query_vector = self.embedder.encode_query(query).tolist()

        # Build filter if provided
        conditions = []
        if ticker:
            conditions.append(
                rest.FieldCondition(
                    key="ticker",
                    match=rest.MatchValue(value=ticker.upper()),
                )
            )
        if chunk_type:
            conditions.append(
                rest.FieldCondition(
                    key="chunk_type",
                    match=rest.MatchValue(value=chunk_type),
                )
            )

        query_filter = rest.Filter(must=conditions) if conditions else None

        # Execute search in Qdrant (using query_points compatible with qdrant-client 1.10+)
        search_res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: List[SearchResult] = []
        for rank, point in enumerate(search_res.points, start=1):
            payload = point.payload or {}
            content = payload.get("content", "")
            snippet = content[:300] + "..." if len(content) > 300 else content

            results.append(
                SearchResult(
                    rank=rank,
                    score=float(point.score),
                    chunk_id=str(payload.get("chunk_id", "")),
                    ticker=str(payload.get("ticker", "")),
                    fiscal_year=int(payload.get("fiscal_year", 2025)),
                    chunk_type=str(payload.get("chunk_type", "")),
                    statement_type=str(payload.get("statement_type", "")),
                    section_title=str(payload.get("section_title", "")),
                    source_pages=list(payload.get("source_pages", [])),
                    content_snippet=snippet,
                    payload=payload,
                )
            )

        return results
