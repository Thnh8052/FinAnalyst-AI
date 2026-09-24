"""Evaluation Benchmark Engine for Multi-Method Financial Retrieval (V1 Hybrid & V0 Dense)."""

import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from financial_rag.config import (
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    EVAL_CUTOFFS,
    OUTPUT_RETRIEVAL,
    OUTPUT_RETRIEVAL_ROOT,
    RETRIEVAL_TOP_K,
    get_collection_name,
    get_provider_config,
)
from financial_rag.retrieval import HybridRetriever, DenseRetriever, SearchResult
from financial_rag.testbed import (
    GoldQuestion,
    TICKER_MAP,
    is_chunk_relevant,
    text_pool_satisfies_signature,
    _get_target_ids,
    normalize_chunk_id,
)
from financial_rag.schema import (
    RETRIEVAL_LEVELS,
    ABSTENTION_LEVEL,
    extract_chunk_text,
)

logger = logging.getLogger(__name__)


def split_by_level(gold_questions: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Partition gold questions into (retrieval_qs, abstention_qs) based on difficulty."""
    retrieval_qs = []
    abstention_qs = []
    for q in gold_questions:
        diff = getattr(q, "difficulty", None) or (q.get("difficulty") if isinstance(q, dict) else "L1")
        if diff in RETRIEVAL_LEVELS:
            retrieval_qs.append(q)
        elif diff == ABSTENTION_LEVEL:
            abstention_qs.append(q)
        else:
            ans = getattr(q, "answerable", True) if not isinstance(q, dict) else q.get("answerable", True)
            (retrieval_qs if ans else abstention_qs).append(q)
    return retrieval_qs, abstention_qs


def compute_ndcg_at_k(binary_relevance: List[int], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    if not any(binary_relevance):
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(binary_relevance[:k]):
        if rel:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(sum(binary_relevance), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def compute_two_tier_retrieval_report(
    method_evals: List[Dict[str, Any]],
    retrieval_qs: List[Any],
    method_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Two-Tier evaluation report."""
    total_q = len(retrieval_qs)
    eval_by_id = {e["query_id"]: e for e in method_evals}

    def _tier_metrics(q_subset: List[Any], label: str) -> Dict[str, Any]:
        n = len(q_subset)
        if n == 0:
            return {"count": 0, "label": label}

        hits = {f"hit@{k}": 0.0 for k in EVAL_CUTOFFS}
        recalls = {f"recall@{k}": 0.0 for k in EVAL_CUTOFFS}
        multi_chunk_hits = {f"multi_chunk_hit@{k}": 0.0 for k in EVAL_CUTOFFS}
        total_mrr = 0.0
        total_ndcg = 0.0
        first_rank_hits = 0.0
        covered_count = 0

        for q in q_subset:
            qid = getattr(q, "id", None) or (q.get("id") if isinstance(q, dict) else "")
            ev = eval_by_id.get(qid)
            if not ev:
                continue

            for k in EVAL_CUTOFFS:
                hits[f"hit@{k}"] += ev.get(f"hit@{k}", ev.get(f"recall@{k}", 0.0))
                recalls[f"recall@{k}"] += ev.get(f"recall@{k}", 0.0)
                multi_chunk_hits[f"multi_chunk_hit@{k}"] += ev.get(f"multi_chunk_hit@{k}", 0.0)

            total_mrr += ev.get("mrr", 0.0)
            total_ndcg += ev.get("ndcg@10", 0.0)
            if ev.get("first_rel_rank") == 1:
                first_rank_hits += 1.0
            if ev.get("coverage", "none") != "none":
                covered_count += 1

        res_dict = {
            "count": n,
            "total_queries": n,
            "valid_queries": n,
            "label": label,
            "coverage_rate": covered_count / n if n > 0 else 0.0,
            "hit@1": first_rank_hits / n if n > 0 else 0.0,
            **{k: v / n for k, v in hits.items()},
            **{k: v / n for k, v in recalls.items()},
            **{k: v / n for k, v in multi_chunk_hits.items()},
            "mrr": total_mrr / n if n > 0 else 0.0,
            "ndcg@10": total_ndcg / n if n > 0 else 0.0,
        }
        return res_dict

    # Tier 1: End-to-End
    tier1 = _tier_metrics(retrieval_qs, "Tier 1: End-to-End Retrieval")

    # Tier 2: Conditional Coverage
    covered_qs = []
    for q in retrieval_qs:
        g_chunks = getattr(q, "gold_chunk_ids", {}) or (q.get("gold_chunk_ids", {}) if isinstance(q, dict) else {})
        if method_key and g_chunks:
            m_chunks = g_chunks.get(method_key, [])
            if isinstance(m_chunks, dict):
                has_ids = bool(m_chunks.get("primary") or m_chunks.get("secondary"))
            elif isinstance(m_chunks, list):
                has_ids = bool(m_chunks)
            else:
                has_ids = False
            if has_ids:
                covered_qs.append(q)
        else:
            qid = getattr(q, "id", None) or (q.get("id") if isinstance(q, dict) else "")
            ev = eval_by_id.get(qid)
            if ev and ev.get("coverage", "none") != "none":
                covered_qs.append(q)

    tier2 = _tier_metrics(covered_qs, "Tier 2: Conditional Retrieval")

    return {
        "tier1_end_to_end": tier1,
        "tier2_conditional": tier2,
        "total_retrieval_queries": total_q,
        "covered_queries_count": len(covered_qs),
    }


def compute_abstention_metrics(
    predictions: List[Dict[str, Any]],
    abstention_qs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute loose and strict abstention accuracy for L5 queries."""
    total = len(abstention_qs)
    if total == 0:
        return {
            "total_abstention_queries": 0,
            "loose_abstention_accuracy": 1.0,
            "strict_abstention_accuracy": 1.0,
        }

    pred_by_id = {p.get("id") or p.get("query_id", ""): p for p in predictions}

    loose_phrases = [
        "cannot", "can not", "not available", "not disclosed", "not disclose", "does not disclose",
        "unanswerable", "abstain", "insufficient", "not provided", "not found", "future",
        "does not contain", "no information", "unable to", "stopped reporting"
    ]

    loose_correct = 0
    strict_correct = 0

    for q in abstention_qs:
        qid = q.get("id") or q.get("query_id", "") if isinstance(q, dict) else getattr(q, "id", "")
        pred = pred_by_id.get(qid, {})
        resp = str(pred.get("response", "") or pred.get("answer", "")).lower()

        # Loose check: contains any refusal/abstention phrase
        is_loose = any(phrase in resp for phrase in loose_phrases)
        if is_loose:
            loose_correct += 1

        # Strict check: also checks context/reason match
        reason = str(q.get("gold_abstention_reason", "") if isinstance(q, dict) else getattr(q, "gold_abstention_reason", "")).lower()
        if is_loose:
            if not reason:
                strict_correct += 1
            else:
                reason_words = [w for w in re.findall(r"\w+", reason) if len(w) > 3]
                if any(w in resp for w in reason_words):
                    strict_correct += 1

    return {
        "total_abstention_queries": total,
        "loose_abstention_accuracy": loose_correct / total if total > 0 else 0.0,
        "strict_abstention_accuracy": strict_correct / total if total > 0 else 0.0,
    }


class RetrievalEvaluator:
    """Runs automated retrieval benchmarks across multiple chunking methods."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        retrieval_mode: str = "hybrid",  # "hybrid" (V1) | "dense" (V0) | "sparse"
        eval_filter_strategy: str = "none",  # "none" = pure search without metadata filter
    ):
        self.retriever = retriever or HybridRetriever()
        self.retrieval_mode = retrieval_mode
        self.eval_filter_strategy = eval_filter_strategy

    def evaluate_query_on_method(
        self,
        gold_q: GoldQuestion,
        method_key: str,
        collection_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        target_ticker = getattr(gold_q, "target_company", "") or ""
        if isinstance(gold_q, dict) and not target_ticker:
            target_ticker = gold_q.get("target_company") or gold_q.get("document", {}).get("company", "")
        target_ticker = TICKER_MAP.get(target_ticker.lower(), target_ticker).upper() if target_ticker else None

        fiscal_year = None
        doc_info = getattr(gold_q, "document", {}) or {}
        if isinstance(gold_q, dict) and not doc_info:
            doc_info = gold_q.get("document", {}) or {}
        if isinstance(doc_info, dict):
            p = str(doc_info.get("reporting_period", ""))
            m_yr = re.search(r"(20\d{2})", p)
            if m_yr:
                fiscal_year = int(m_yr.group(1))

        use_filters = self.eval_filter_strategy != "none"

        results: List[SearchResult] = self.retriever.search(
            collection_name=collection_name,
            query=gold_q.question,
            top_k=top_k,
            mode=self.retrieval_mode,
            ticker=(target_ticker if use_filters and target_ticker != "MULTI" else None),
            fiscal_year=(fiscal_year if use_filters else None),
            filter_strategy=self.eval_filter_strategy if use_filters else "none",
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        binary_relevance: List[int] = []
        first_rel_rank: Optional[int] = None
        for res in results:
            rel = 1 if is_chunk_relevant(res.payload, gold_q, method_key) else 0
            binary_relevance.append(rel)
            if rel == 1 and first_rel_rank is None:
                first_rel_rank = res.rank

        # Academic Distinction: Hit@k (Success@k) vs Recall@k (Proportion of ground truth retrieved)
        hits = {}
        for k in EVAL_CUTOFFS:
            hits[f"hit@{k}"] = 1.0 if any(binary_relevance[:k]) else 0.0

        target_ids = _get_target_ids(gold_q, method_key)
        recalls = {}
        if target_ids:
            norm_targets = {normalize_chunk_id(x) for x in target_ids}
            num_targets = len(norm_targets)
            for k in EVAL_CUTOFFS:
                retrieved_target_matches = {
                    normalize_chunk_id(r.chunk_id)
                    for i, r in enumerate(results[:k])
                    if binary_relevance[i] == 1 and normalize_chunk_id(r.chunk_id) in norm_targets
                }
                if retrieved_target_matches:
                    recalls[f"recall@{k}"] = len(retrieved_target_matches) / num_targets
                elif any(binary_relevance[:k]):
                    # Relevant chunk matched via content signature
                    recalls[f"recall@{k}"] = min(1.0, sum(binary_relevance[:k]) / num_targets)
                else:
                    recalls[f"recall@{k}"] = 0.0
        else:
            for k in EVAL_CUTOFFS:
                recalls[f"recall@{k}"] = hits[f"hit@{k}"]

        multi_chunk_hits = {}
        for k in EVAL_CUTOFFS:
            pool_text = " ".join(extract_chunk_text(r.payload) for r in results[:k])
            multi_chunk_hits[f"multi_chunk_hit@{k}"] = (
                1.0 if text_pool_satisfies_signature(gold_q, pool_text) else 0.0
            )

        mrr = 1.0 / first_rel_rank if first_rel_rank is not None else 0.0
        ndcg_10 = compute_ndcg_at_k(binary_relevance, 10)

        method_chunk_ids = gold_q.gold_chunk_ids.get(method_key, [])
        if isinstance(method_chunk_ids, dict):
            has_evidence = bool(method_chunk_ids.get("primary") or method_chunk_ids.get("secondary"))
        elif isinstance(method_chunk_ids, list):
            has_evidence = bool(method_chunk_ids)
        else:
            has_evidence = False
        coverage = "full" if has_evidence else "none"

        return {
            "query_id": gold_q.id,
            "difficulty": gold_q.difficulty,
            "question_type": gold_q.question_type,
            "target_company": gold_q.target_company,
            "coverage": coverage,
            "latency_ms": latency_ms,
            "first_rel_rank": first_rel_rank,
            "mrr": mrr,
            "ndcg@10": ndcg_10,
            **hits,
            **recalls,
            **multi_chunk_hits,
            "top_hits": [
                {
                    "rank": r.rank,
                    "score": round(r.score, 4),
                    "chunk_id": r.chunk_id,
                    "is_relevant": bool(binary_relevance[i]),
                    "chunk_type": r.chunk_type,
                    "section_title": r.section_title,
                }
                for i, r in enumerate(results[:5])
            ],
        }

    def run_benchmark(
        self,
        gold_questions: List[GoldQuestion],
        methods: Optional[List[str]] = None,
        model_key: str = DEFAULT_EMBEDDING_PROVIDER,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        target_methods = methods or list(CHUNK_METHODS.keys())
        
        # Determine output directory
        if output_dir is None:
            if self.retrieval_mode == "hybrid":
                output_dir = OUTPUT_RETRIEVAL / "v1_hybrid"
            elif self.retrieval_mode == "dense":
                output_dir = OUTPUT_RETRIEVAL / "v0_dense_baseline"
            else:
                output_dir = OUTPUT_RETRIEVAL / f"v1_{self.retrieval_mode}"
        output_dir.mkdir(parents=True, exist_ok=True)

        retrieval_qs, abstention_qs = split_by_level(gold_questions)
        logger.info(
            f"Split gold set: {len(retrieval_qs)} Retrieval (L1-L4) | "
            f"{len(abstention_qs)} Abstention (L5)"
        )

        if target_methods and retrieval_qs:
            warmup_coll = get_collection_name(model_key, target_methods[0])
            logger.info(f"Warming up retriever ({self.retrieval_mode}) with 3 test queries...")
            for wq in retrieval_qs[:3]:
                _ = self.retriever.search(
                    collection_name=warmup_coll,
                    query=wq.question,
                    top_k=5,
                    mode=self.retrieval_mode,
                )

        benchmark_results = {}

        for method_key in target_methods:
            method_info = CHUNK_METHODS[method_key]
            coll_name = get_collection_name(model_key, method_key)
            disp_name = method_info["display_name"]
            logger.info(f"Evaluating {disp_name} [{self.retrieval_mode}] (collection: {coll_name})...")

            method_evals = [
                self.evaluate_query_on_method(gold_q, method_key, coll_name)
                for gold_q in retrieval_qs
            ]

            if not method_evals:
                continue

            two_tier = compute_two_tier_retrieval_report(method_evals, retrieval_qs, method_key)

            by_diff = defaultdict(list)
            for e in method_evals:
                by_diff[e["difficulty"]].append(e)

            diff_breakdown = {}
            for diff_k in sorted(by_diff.keys()):
                q_list = by_diff[diff_k]
                n = len(q_list)
                diff_breakdown[diff_k] = {
                    "count": n,
                    "hit@1": sum(1.0 for x in q_list if x.get("first_rel_rank") == 1) / n,
                    "hit@5": sum(x.get("hit@5", x.get("recall@5", 0.0)) for x in q_list) / n,
                    "recall@5": sum(x["recall@5"] for x in q_list) / n,
                    "recall@10": sum(x["recall@10"] for x in q_list) / n,
                    "multi_chunk_hit@5": sum(x["multi_chunk_hit@5"] for x in q_list) / n,
                    "mrr": sum(x["mrr"] for x in q_list) / n,
                }

            by_type = defaultdict(list)
            for e in method_evals:
                by_type[e["question_type"]].append(e)

            type_breakdown = {}
            for q_type, q_list in by_type.items():
                n = len(q_list)
                type_breakdown[q_type] = {
                    "count": n,
                    "hit@5": sum(x.get("hit@5", x.get("recall@5", 0.0)) for x in q_list) / n,
                    "recall@5": sum(x["recall@5"] for x in q_list) / n,
                    "recall@10": sum(x["recall@10"] for x in q_list) / n,
                    "mrr": sum(x["mrr"] for x in q_list) / n,
                }

            by_comp = defaultdict(list)
            for e in method_evals:
                by_comp[e["target_company"]].append(e)

            comp_breakdown = {}
            for comp_k, q_list in sorted(by_comp.items()):
                n = len(q_list)
                comp_breakdown[comp_k] = {
                    "count": n,
                    "hit@5": sum(x.get("hit@5", x.get("recall@5", 0.0)) for x in q_list) / n,
                    "recall@5": sum(x["recall@5"] for x in q_list) / n,
                    "recall@10": sum(x["recall@10"] for x in q_list) / n,
                    "mrr": sum(x["mrr"] for x in q_list) / n,
                }

            benchmark_results[method_key] = {
                "display_name": disp_name,
                "collection_name": coll_name,
                "mode": self.retrieval_mode,
                "overall": two_tier["tier1_end_to_end"],
                "tier2_conditional": two_tier["tier2_conditional"],
                "by_difficulty": diff_breakdown,
                "by_question_type": type_breakdown,
                "by_company": comp_breakdown,
                "abstention_queries_count": len(abstention_qs),
                "details": method_evals,
            }

        # Save summary JSON
        summary_name = "v1_hybrid_evaluation_summary.json" if self.retrieval_mode == "hybrid" else "v0_retrieval_evaluation_summary.json"
        summary_path = output_dir / summary_name
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary JSON saved to {summary_path}")

        # Save Markdown Report
        report_md = self.generate_markdown_report(benchmark_results, abstention_qs, model_key=model_key, mode=self.retrieval_mode)
        report_names = (
            ("V1_HYBRID_RETRIEVAL_REPORT.md", "hybrid_retrieval_report.md")
            if self.retrieval_mode == "hybrid"
            else ("V0_RETRIEVAL_EVALUATION_REPORT.md", "dense_baseline_report.md")
        )
        for name in report_names:
            path = output_dir / name
            with open(path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"Report saved to {path}")

        return benchmark_results

    @staticmethod
    def generate_markdown_report(
        benchmark_results: Dict[str, Any],
        abstention_qs: Optional[List[Any]] = None,
        model_key: str = DEFAULT_EMBEDDING_PROVIDER,
        mode: str = "hybrid",
    ) -> str:
        provider = get_provider_config(model_key) if model_key in EMBEDDING_PROVIDERS else {"model_name": model_key, "embedding_dim": 768}
        mode_title = "V1 Native Hybrid (Dense + Qdrant/BM25 RRF)" if mode == "hybrid" else f"V0 {mode.capitalize()}-Only"
        lines = [
            f"# 📊 {mode_title} Retrieval Evaluation Report",
            "",
            "**FinAnalyst-AI: Multi-Method Financial Chunking & Retrieval Benchmark**",
            "",
            f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Embedding Model:** `{provider.get('model_name', model_key)}` ({provider.get('embedding_dim', 768)}-dim)",
            f"- **Sparse Model:** `Qdrant/bm25` (FastEmbed native lexical scoring)" if mode == "hybrid" else "",
            f"- **Fusion Algorithm:** Reciprocal Rank Fusion (RRF, k=60)" if mode == "hybrid" else "",
            f"- **Evaluation Mode:** {mode_title}",
            "",
            "## 1. Tier 1 — End-to-End Retrieval (All Retrieval Queries)",
            "",
            "| Method | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Multi-Chunk@5 | Multi-Chunk@10 | MRR | NDCG@10 | Avg Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        # Filter out empty lines
        lines = [l for l in lines if l != ""]

        for m_key, res in benchmark_results.items():
            ov = res["overall"]
            avg_lat = sum(x["latency_ms"] for x in res.get("details", [])) / len(res.get("details", [1]))
            lines.append(
                f"| **{res['display_name']}** | {ov.get('hit@1', 0.0):.1%} | {ov.get('hit@5', ov.get('recall@5', 0.0)):.1%} | "
                f"{ov.get('recall@5', 0.0):.1%} | {ov.get('recall@10', 0.0):.1%} | {ov.get('multi_chunk_hit@5', 0.0):.1%} | "
                f"{ov.get('multi_chunk_hit@10', 0.0):.1%} | {ov.get('mrr', 0.0):.3f} | {ov.get('ndcg@10', 0.0):.3f} | "
                f"{avg_lat:.1f}ms |"
            )

        lines.extend([
            "",
            "## 2. Tier 2 — Conditional Retrieval (Coverage != 'none')",
            "",
            "| Method | Covered Queries | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Multi-Chunk@5 | MRR | NDCG@10 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for m_key, res in benchmark_results.items():
            t2 = res["tier2_conditional"]
            lines.append(
                f"| **{res['display_name']}** | {t2.get('count', 0)} ({t2.get('coverage_rate', 0.0):.1%}) | "
                f"{t2.get('hit@1', 0.0):.1%} | {t2.get('hit@5', t2.get('recall@5', 0.0)):.1%} | {t2.get('recall@5', 0.0):.1%} | "
                f"{t2.get('recall@10', 0.0):.1%} | {t2.get('multi_chunk_hit@5', 0.0):.1%} | {t2.get('mrr', 0.0):.3f} | {t2.get('ndcg@10', 0.0):.3f} |"
            )

        lines.extend([
            "",
            "## 3. Performance Breakdown by Difficulty Level (L1 - L4)",
            "",
        ])

        for m_key, res in benchmark_results.items():
            lines.append(f"### {res['display_name']}")
            lines.append("| Level | Count | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Multi-Chunk@5 | MRR |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for diff, stats in res.get("by_difficulty", {}).items():
                lines.append(
                    f"| **{diff}** | {stats['count']} | {stats.get('hit@1', 0.0):.1%} | "
                    f"{stats.get('hit@5', stats.get('recall@5', 0.0)):.1%} | {stats.get('recall@5', 0.0):.1%} | "
                    f"{stats.get('recall@10', 0.0):.1%} | {stats.get('multi_chunk_hit@5', 0.0):.1%} | {stats.get('mrr', 0.0):.3f} |"
                )
            lines.append("")

        if abstention_qs:
            lines.extend([
                "## 4. Abstention (L5 Out-of-Scope / Non-Answerable Queries)",
                "",
                f"- Total L5 Queries: **{len(abstention_qs)}**",
                "- Evaluated separately — excluded from standard retrieval recall denominators.",
                "",
            ])

        lines.extend([
            "## 5. Ghi Chú Phương Pháp Luận & Chuẩn Mực Học Thuật",
            "",
            "- **Hit@k (Success@k)**: Tỷ lệ câu hỏi tìm thấy ít nhất 1 chunk bằng chứng liên quan trong Top-k ($|\\text{Retrieved@k} \\cap \\text{Gold}| \\ge 1$).",
            "- **Recall@k (Academic Set-Based Recall)**: Tỷ lệ chunk bằng chứng truy xuất được trên tổng số chunk Ground-Truth yêu cầu ($\\frac{|\\text{Retrieved@k} \\cap \\text{Gold}|}{|\\text{Gold}|}$). Với L1/L2/L4 (1 chunk), $\\text{Recall@k} = \\text{Hit@k}$. Với L3 (so sánh liên kỳ/đa bảng yêu cầu $M \\ge 2$ chunks), nếu chỉ lấy trúng 1 chunk thì $\\text{Recall@k} = 50\\%$.",
            "- **Multi-Chunk@k**: Tỷ lệ truy vấn mà văn bản gộp (text pool) của Top-k chunks cùng lúc thỏa mãn toàn bộ chữ ký tài chính (financial facts & keywords).",
            "- **MRR & NDCG@10**: Đánh giá vị trí xuất hiện của chunk đầu tiên và độ suy giảm vị trí của các chunk liên quan.",
            "",
        ])

        lines.append("---")
        lines.append("*Auto-generated by FinAnalyst-AI V1 Hybrid Retrieval Benchmark Engine.*")

        return "\n".join(lines)


# Backward compatibility
evaluate_retrieval = compute_two_tier_retrieval_report
