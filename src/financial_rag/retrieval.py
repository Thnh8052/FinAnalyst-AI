"""Unified Hybrid Retrieval Engine for Qdrant Vector Collections (V1 Architecture).

Combines:
1. Dense Semantic Search: BGE-base-en-v1.5 (768-dim, Cosine) for high-recall conceptual matching.
2. Sparse Lexical Search: Qdrant Native BM25 via FastEmbed ('Qdrant/bm25') for exact financial terms & entities.
3. Database-Level Fusion: Reciprocal Rank Fusion (RRF, k=60) computed natively in Qdrant engine.
4. Auto-escalation: Selectivity-aware pre-filtering to prevent empty candidate pools on narrow financial filters.
5. Persistent Query Caching: Instant 0ms embedding latency for previously evaluated benchmark queries.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from financial_rag.config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DENSE_PREFETCH_DEPTH,
    FILTER_STRATEGY,
    PREFETCH_RETRY_DEPTH,
    QDRANT_PREFER_LOCAL,
    QDRANT_STORAGE_PATH,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
)
from financial_rag.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual chunk search hit from Qdrant Hybrid/Dense/Sparse Retriever."""
    rank: int
    score: float
    chunk_id: str
    ticker: str
    fiscal_year: int
    fiscal_period: str
    chunk_type: str
    statement_type: str
    section_title: str
    source_pages: List[int]
    contains_table: bool
    contains_text: bool
    content_snippet: str
    payload: Dict[str, Any]


_SELECTIVE_FILTER_FIELDS = ("ticker", "fiscal_year", "statement_type", "document_id")


class HybridRetriever:
    """Performs Hybrid Search (Dense BGE-base + Sparse Qdrant/BM25 + RRF) against Qdrant collections."""

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
        prefer_local: bool = QDRANT_PREFER_LOCAL,
        model_key: str = DEFAULT_EMBEDDING_PROVIDER,
        embedder: Optional[EmbeddingEngine] = None,
        sparse_model_name: str = "Qdrant/bm25",
        cache_path: Optional[Union[str, Path]] = None,
    ):
        if prefer_local:
            target_path = Path(storage_path or QDRANT_STORAGE_PATH)
            logger.info(f"Connecting HybridRetriever to Local Qdrant: {target_path}")
            self.client = QdrantClient(path=str(target_path))
            self.is_local = True
        else:
            target_url = url or QDRANT_URL
            logger.info(f"Connecting HybridRetriever to Qdrant URL: {target_url}")
            self.client = QdrantClient(url=target_url)
            self.is_local = False

        self.model_key = model_key
        self.embedder = embedder or EmbeddingEngine.get_instance(model_key)
        self.sparse_model_name = sparse_model_name
        self._sparse_embedder = None

        # Persistent query embedding cache
        self._cache_path = Path(cache_path) if cache_path else Path("data/embeddings_cache/query_cache.json")
        self._dense_cache: Dict[str, List[float]] = {}
        self._sparse_cache: Dict[str, Dict[str, List[Any]]] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Query Embedding & Cache Management
    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._dense_cache = data.get("dense", {})
                    self._sparse_cache = data.get("sparse", {})
                logger.info(
                    f"Loaded {len(self._dense_cache)} dense and {len(self._sparse_cache)} sparse cached query vectors from {self._cache_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load query cache from {self._cache_path}: {e}")

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump({"dense": self._dense_cache, "sparse": self._sparse_cache}, f)
            logger.info(f"Saved query cache to {self._cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save query cache to {self._cache_path}: {e}")

    def _get_dense_vector(self, query: str) -> List[float]:
        # FIX [B5]: Isolate cache key by model_key to prevent cross-model vector pollution
        cache_key = f"{self.model_key}::{query}"
        if cache_key in self._dense_cache:
            return self._dense_cache[cache_key]
        vec = self.embedder.encode_query(query).tolist()
        self._dense_cache[cache_key] = vec
        return vec

    def _get_sparse_vector(self, query: str) -> rest.SparseVector:
        # FIX [B5]: Isolate sparse cache key by sparse_model_name
        cache_key = f"{self.sparse_model_name}::{query}"
        if cache_key in self._sparse_cache:
            cached = self._sparse_cache[cache_key]
            return rest.SparseVector(indices=cached["indices"], values=cached["values"])

        if self._sparse_embedder is None:
            try:
                from fastembed import SparseTextEmbedding
                self._sparse_embedder = SparseTextEmbedding(model_name=self.sparse_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize FastEmbed SparseTextEmbedding: {e}")
                raise

        sparse_results = list(self._sparse_embedder.embed([query]))
        s_vec = sparse_results[0]
        indices = s_vec.indices.tolist()
        values = s_vec.values.tolist()
        self._sparse_cache[cache_key] = {"indices": indices, "values": values}
        return rest.SparseVector(indices=indices, values=values)

    # ------------------------------------------------------------------
    # Filtering & Selection Helpers
    # ------------------------------------------------------------------
    def _matches_filters(
        self,
        payload: Dict[str, Any],
        ticker: Optional[str] = None,
        document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        statement_type: Optional[str] = None,
        form_type: Optional[str] = None,
        contains_table: Optional[bool] = None,
        contains_text: Optional[bool] = None,
        simplified_mode: bool = True,
    ) -> bool:
        if ticker and str(payload.get("ticker", "")).upper() != ticker.upper():
            return False
        if document_id and str(payload.get("document_id", "")).lower() != document_id.lower():
            return False

        # Period year matching (bảo toàn mảng period_years cho năm quá khứ & so sánh liên kỳ)
        target_year = period_year if period_year is not None else fiscal_year
        if target_year is not None:
            chunk_period_years = payload.get("period_years") or []
            chunk_years = {int(y) for y in chunk_period_years if str(y).isdigit()}
            filing_year = int(payload.get("fiscal_year", 0))
            if chunk_years:
                if int(target_year) not in chunk_years and filing_year != int(target_year):
                    return False
            else:
                if filing_year != int(target_year):
                    return False

        if form_type and str(payload.get("form_type", "")) != form_type:
            return False

        # Các trường tùy chọn (giản lược theo sec-insights: không ép buộc lọc nếu không truyền)
        if fiscal_period and str(payload.get("fiscal_period", "")).upper() != fiscal_period.upper():
            return False
        if statement_type and str(payload.get("statement_type", "")) != statement_type:
            return False
        if contains_table is not None and bool(payload.get("contains_table")) != bool(contains_table):
            return False
        if contains_text is not None and bool(payload.get("contains_text")) != bool(contains_text):
            return False
        return True

    @staticmethod
    def _is_selective(
        ticker: Optional[str],
        fiscal_year: Optional[int],
        statement_type: Optional[str],
        document_id: Optional[str] = None,
    ) -> bool:
        selective_count = sum([
            ticker is not None,
            fiscal_year is not None,
            statement_type is not None,
            document_id is not None,
        ])
        return selective_count >= 2

    def _build_filter(
        self,
        ticker: Optional[str] = None,
        document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        statement_type: Optional[str] = None,
        form_type: Optional[str] = None,
        contains_table: Optional[bool] = None,
        contains_text: Optional[bool] = None,
        simplified_mode: bool = True,
    ) -> Optional[rest.Filter]:
        conditions = []
        if ticker:
            conditions.append(rest.FieldCondition(key="ticker", match=rest.MatchValue(value=ticker.upper())))
        if document_id:
            conditions.append(rest.FieldCondition(key="document_id", match=rest.MatchValue(value=document_id.lower())))

        target_year = period_year if period_year is not None else fiscal_year
        if target_year is not None:
            # Khớp nếu target_year nằm trong mảng period_years HOẶC là fiscal_year của filing
            conditions.append(
                rest.Filter(
                    should=[
                        rest.FieldCondition(key="period_years", match=rest.MatchValue(value=int(target_year))),
                        rest.FieldCondition(key="fiscal_year", match=rest.MatchValue(value=int(target_year))),
                    ]
                )
            )

        if form_type:
            conditions.append(rest.FieldCondition(key="form_type", match=rest.MatchValue(value=form_type)))
        if fiscal_period:
            conditions.append(rest.FieldCondition(key="fiscal_period", match=rest.MatchValue(value=fiscal_period)))
        if statement_type:
            conditions.append(rest.FieldCondition(key="statement_type", match=rest.MatchValue(value=statement_type)))
        if contains_table is not None:
            conditions.append(rest.FieldCondition(key="contains_table", match=rest.MatchValue(value=bool(contains_table))))
        if contains_text is not None:
            conditions.append(rest.FieldCondition(key="contains_text", match=rest.MatchValue(value=bool(contains_text))))
        return rest.Filter(must=conditions) if conditions else None

    # ------------------------------------------------------------------
    # Unified Search API (Hybrid / Dense / Sparse)
    # ------------------------------------------------------------------
    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        mode: str = "hybrid",  # "hybrid" (V1) | "dense" (V0 baseline) | "sparse" (Pure BM25)
        filter_strategy: str = FILTER_STRATEGY,
        prefetch_depth: int = DENSE_PREFETCH_DEPTH,
        ticker: Optional[str] = None,
        document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        period_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        statement_type: Optional[str] = None,
        form_type: Optional[str] = None,
        contains_table: Optional[bool] = None,
        contains_text: Optional[bool] = None,
        simplified_mode: bool = True,
    ) -> List[SearchResult]:
        """Perform search across Qdrant collections.
        
        Modes:
        - 'hybrid': Native Dual Prefetch (Dense + BM25) fused via Reciprocal Rank Fusion (RRF).
        - 'dense': Pure Cosine Dense search (V0 baseline).
        - 'sparse': Pure BM25 Sparse Lexical search.
        """
        has_filters = any([
            ticker is not None,
            document_id is not None,
            fiscal_year is not None,
            period_year is not None,
            fiscal_period is not None,
            statement_type is not None,
            form_type is not None,
            contains_table is not None,
            contains_text is not None,
        ])

        # Auto-escalate to pre_filter if filter set is selective
        effective_strategy = filter_strategy
        if (
            filter_strategy == "post_filter"
            and has_filters
            and self._is_selective(ticker, fiscal_year, statement_type, document_id)
        ):
            effective_strategy = "pre_filter"

        q_filter = (
            self._build_filter(
                ticker=ticker,
                document_id=document_id,
                fiscal_year=fiscal_year,
                period_year=period_year,
                fiscal_period=fiscal_period,
                statement_type=statement_type,
                form_type=form_type,
                contains_table=contains_table,
                contains_text=contains_text,
                simplified_mode=simplified_mode,
            )
            if (effective_strategy == "pre_filter" and has_filters)
            else None
        )

        # ---------------------------------------------------------
        # Branch 1: Hybrid Search (Dense + Qdrant/BM25 via RRF)
        # ---------------------------------------------------------
        if mode == "hybrid":
            query_dense_vec = self._get_dense_vector(query)
            query_sparse_vec = self._get_sparse_vector(query)
            fetch_limit = max(prefetch_depth, top_k * 2)
            is_post_filter = (effective_strategy == "post_filter" and has_filters)
            fusion_limit = fetch_limit if is_post_filter else top_k

            try:
                search_res = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        rest.Prefetch(
                            query=query_dense_vec,
                            using="dense",
                            limit=fetch_limit,
                            filter=q_filter,
                        ),
                        rest.Prefetch(
                            query=query_sparse_vec,
                            using="sparse",
                            limit=fetch_limit,
                            filter=q_filter,
                        ),
                    ],
                    query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                    limit=fusion_limit,
                    with_payload=True,
                )
                raw_points = search_res.points
            except Exception as e:
                # Graceful fallback to dense if sparse vector index is unpopulated
                logger.warning(f"Hybrid search failed ({e}). Falling back to pure dense search.")
                search_res = self.client.query_points(
                    collection_name=collection_name,
                    query=query_dense_vec,
                    using="dense",
                    query_filter=q_filter,
                    limit=fusion_limit,
                    with_payload=True,
                )
                raw_points = search_res.points

            if is_post_filter:
                filtered_points = [
                    pt for pt in raw_points
                    if self._matches_filters(
                        pt.payload or {},
                        ticker=ticker,
                        document_id=document_id,
                        fiscal_year=fiscal_year,
                        period_year=period_year,
                        fiscal_period=fiscal_period,
                        statement_type=statement_type,
                        form_type=form_type,
                        contains_table=contains_table,
                        contains_text=contains_text,
                        simplified_mode=simplified_mode,
                    )
                ]
                points_to_use = filtered_points[:top_k]
            else:
                points_to_use = raw_points[:top_k]

        # ---------------------------------------------------------
        # Branch 2: Sparse Lexical Search (Pure BM25)
        # ---------------------------------------------------------
        elif mode == "sparse":
            query_sparse_vec = self._get_sparse_vector(query)
            fetch_limit = max(prefetch_depth, top_k * 2)
            is_post_filter = (effective_strategy == "post_filter" and has_filters)
            search_limit = fetch_limit if is_post_filter else top_k

            search_res = self.client.query_points(
                collection_name=collection_name,
                query=query_sparse_vec,
                using="sparse",
                query_filter=q_filter,
                limit=search_limit,
                with_payload=True,
            )
            if is_post_filter:
                filtered_points = [
                    pt for pt in search_res.points
                    if self._matches_filters(
                        pt.payload or {},
                        ticker=ticker,
                        document_id=document_id,
                        fiscal_year=fiscal_year,
                        period_year=period_year,
                        fiscal_period=fiscal_period,
                        statement_type=statement_type,
                        form_type=form_type,
                        contains_table=contains_table,
                        contains_text=contains_text,
                        simplified_mode=simplified_mode,
                    )
                ]
                points_to_use = filtered_points[:top_k]
            else:
                points_to_use = search_res.points[:top_k]

        # ---------------------------------------------------------
        # Branch 3: Dense Semantic Search (V0 Baseline)
        # ---------------------------------------------------------
        else:
            query_dense_vec = self._get_dense_vector(query)

            if effective_strategy == "post_filter" and has_filters:
                fetch_limit = max(prefetch_depth, top_k)
                search_res = self.client.query_points(
                    collection_name=collection_name,
                    query=query_dense_vec,
                    using="dense",
                    query_filter=None,
                    limit=fetch_limit,
                    with_payload=True,
                )
                filtered_points = [
                    pt for pt in search_res.points
                    if self._matches_filters(
                        pt.payload or {},
                        ticker=ticker,
                        document_id=document_id,
                        fiscal_year=fiscal_year,
                        period_year=period_year,
                        fiscal_period=fiscal_period,
                        statement_type=statement_type,
                        form_type=form_type,
                        contains_table=contains_table,
                        contains_text=contains_text,
                        simplified_mode=simplified_mode,
                    )
                ]

                # Cap retry at fetch_limit * 4 to prevent runaway fetch
                if not filtered_points and fetch_limit < PREFETCH_RETRY_DEPTH:
                    retry_limit = min(PREFETCH_RETRY_DEPTH, fetch_limit * 4)
                    search_res_retry = self.client.query_points(
                        collection_name=collection_name,
                        query=query_dense_vec,
                        using="dense",
                        query_filter=None,
                        limit=retry_limit,
                        with_payload=True,
                    )
                    filtered_points = [
                        pt for pt in search_res_retry.points
                        if self._matches_filters(
                            pt.payload or {},
                            ticker=ticker,
                            document_id=document_id,
                            fiscal_year=fiscal_year,
                            period_year=period_year,
                            fiscal_period=fiscal_period,
                            statement_type=statement_type,
                            form_type=form_type,
                            contains_table=contains_table,
                            contains_text=contains_text,
                            simplified_mode=simplified_mode,
                        )
                    ]
                points_to_use = filtered_points[:top_k]

            else:
                # Pre-filter or no filter
                search_res = self.client.query_points(
                    collection_name=collection_name,
                    query=query_dense_vec,
                    using="dense",
                    query_filter=q_filter,
                    limit=top_k,
                    with_payload=True,
                )
                points_to_use = search_res.points

        # Format output
        results: List[SearchResult] = []
        for rank, point in enumerate(points_to_use, start=1):
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
                    fiscal_period=str(payload.get("fiscal_period", "FY")),
                    chunk_type=str(payload.get("chunk_type", "")),
                    statement_type=str(payload.get("statement_type", "")),
                    section_title=str(payload.get("section_title", "")),
                    source_pages=list(payload.get("source_pages", [])),
                    contains_table=bool(payload.get("contains_table", False)),
                    contains_text=bool(payload.get("contains_text", False)),
                    content_snippet=snippet,
                    payload=payload,
                )
            )

        return results

    def close(self) -> None:
        """Persist query embedding cache and close connection."""
        self._save_cache()
        if hasattr(self.client, "close"):
            self.client.close()


# Backward compatibility alias
DenseRetriever = HybridRetriever
