"""Fast Comprehensive Diagnostic Script for Financial RAG Retrieval Performance (Method 5 & Cross-Method).

Answers:
1. Are GT chunk_ids valid in Qdrant? (Exact vs Normalized vs Missing)
2. Where do missed GT chunks end up? (Rank 11-20, 21-50, 51-100, or >100)
3. Do competing Top-3 chunks actually contain the ground truth numbers/facts?
4. How do different difficulty levels (L1, L2, L3, L4) behave?
"""

import sys
import re
import json
from collections import defaultdict
from pathlib import Path

from financial_rag.config import (
    GOLD_TEST_SET_FILE,
    get_collection_name,
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    FILTER_STRATEGY,
)
from financial_rag.testbed import (
    load_gold_test_set,
    normalize_chunk_id,
    is_chunk_relevant,
    TICKER_MAP,
)
from financial_rag.retrieval import DenseRetriever
from financial_rag.schema import RETRIEVAL_LEVELS, extract_chunk_text


def get_all_collection_chunk_ids(client, coll_name):
    """Retrieve all chunk_ids from a collection into a set in memory quickly."""
    all_raw = set()
    all_norm = {}  # norm_id -> raw_id
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=coll_name,
            scroll_filter=None,
            limit=2000,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for p in pts:
            cid = p.payload.get("chunk_id", "")
            if cid:
                all_raw.add(cid)
                all_norm[normalize_chunk_id(cid)] = cid
        if offset is None:
            break
    return all_raw, all_norm


def run_diagnostic():
    print("=" * 80)
    print("FINANCIAL RAG RETRIEVAL DIAGNOSTIC REPORT (DEEP ROOT CAUSE ANALYSIS)")
    print("=" * 80)

    # 1. Load Gold Test Set
    questions = load_gold_test_set(GOLD_TEST_SET_FILE)
    retrieval_qs = [q for q in questions if q.difficulty in RETRIEVAL_LEVELS]
    print(f"Total Questions: {len(questions)} | Retrieval Questions (L1-L4): {len(retrieval_qs)}")

    retriever = DenseRetriever()

    m5_key = "method5_proposed_golden_hybrid"
    m5_coll = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, m5_key)

    print(f"\n--- PART 1: GT CHUNK ID VALIDITY CHECK IN QDRANT (126 Queries x 5 Methods) ---")
    validity_stats = {}
    for m_key in CHUNK_METHODS.keys():
        coll = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, m_key)
        all_raw, all_norm = get_all_collection_chunk_ids(retriever.client, coll)

        exact_ok = 0
        norm_ok = 0
        missing = 0
        no_gt = 0

        for q in retrieval_qs:
            g_ids = q.gold_chunk_ids.get(m_key, [])
            if not g_ids:
                no_gt += 1
                continue
            raw_id = g_ids[0]
            norm_id = normalize_chunk_id(raw_id)

            if raw_id in all_raw:
                exact_ok += 1
            elif norm_id in all_norm:
                norm_ok += 1
            else:
                missing += 1

        validity_stats[m_key] = {
            "exact_ok": exact_ok,
            "norm_ok": norm_ok,
            "missing": missing,
            "no_gt": no_gt,
            "total_valid": exact_ok + norm_ok,
        }
        print(f"Method: {m_key:<32} | Exact: {exact_ok:>3}/126 | Norm-Recovered: {norm_ok:>3}/126 | Missing/Broken: {missing:>3}/126")

    print(f"\n--- PART 2: METHOD 5 DEEP RANKING DISTRIBUTION (TOP-100 SEARCH) ---")
    rank_dist = {
        "rank_1": 0,
        "rank_2_5": 0,
        "rank_6_10": 0,
        "rank_11_20": 0,
        "rank_21_50": 0,
        "rank_51_100": 0,
        "not_in_top_100": 0,
    }
    by_diff_ranks = defaultdict(lambda: defaultdict(int))
    miss_details = []

    for idx, q in enumerate(retrieval_qs):
        target_ticker = getattr(q, "target_company", "") or ""
        target_ticker = TICKER_MAP.get(target_ticker.lower(), target_ticker).upper() if target_ticker else None

        fiscal_year = None
        doc_info = getattr(q, "document", {}) or {}
        if isinstance(doc_info, dict):
            p = str(doc_info.get("reporting_period", ""))
            m_yr = re.search(r"(20\d{2})", p)
            if m_yr:
                fiscal_year = int(m_yr.group(1))

        # Search top 100
        results = retriever.search(
            collection_name=m5_coll,
            query=q.question,
            top_k=100,
            ticker=target_ticker if target_ticker != "MULTI" else None,
            fiscal_year=fiscal_year,
            filter_strategy=FILTER_STRATEGY,
        )

        gt_ids = q.gold_chunk_ids.get(m5_key, [])
        norm_gt_ids = {normalize_chunk_id(x) for x in gt_ids}

        found_rank = None
        for r in results:
            if normalize_chunk_id(r.chunk_id) in norm_gt_ids or is_chunk_relevant(r.payload, q, m5_key):
                found_rank = r.rank
                break

        diff = q.difficulty
        if found_rank == 1:
            rank_dist["rank_1"] += 1
            by_diff_ranks[diff]["rank_1"] += 1
        elif found_rank is not None and 2 <= found_rank <= 5:
            rank_dist["rank_2_5"] += 1
            by_diff_ranks[diff]["rank_2_5"] += 1
        elif found_rank is not None and 6 <= found_rank <= 10:
            rank_dist["rank_6_10"] += 1
            by_diff_ranks[diff]["rank_6_10"] += 1
        elif found_rank is not None and 11 <= found_rank <= 20:
            rank_dist["rank_11_20"] += 1
            by_diff_ranks[diff]["rank_11_20"] += 1
        elif found_rank is not None and 21 <= found_rank <= 50:
            rank_dist["rank_21_50"] += 1
            by_diff_ranks[diff]["rank_21_50"] += 1
        elif found_rank is not None and 51 <= found_rank <= 100:
            rank_dist["rank_51_100"] += 1
            by_diff_ranks[diff]["rank_51_100"] += 1
        else:
            rank_dist["not_in_top_100"] += 1
            by_diff_ranks[diff]["not_in_top_100"] += 1

        # Record miss if not in Top 10
        if found_rank is None or found_rank > 10:
            top_competitors = []
            for r in results[:3]:
                c_text = extract_chunk_text(r.payload)
                has_kws = any(kw.lower() in c_text.lower() for kw in q.keywords) if q.keywords else False
                top_competitors.append({
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "section": r.section_title,
                    "has_target_keywords": has_kws,
                })
            miss_details.append({
                "qid": q.id,
                "difficulty": q.difficulty,
                "target_company": q.target_company,
                "question": q.question,
                "gt_ids": gt_ids,
                "found_rank": found_rank,
                "top_competitors": top_competitors,
            })

    total_r = len(retrieval_qs)
    hit_10_total = rank_dist['rank_1'] + rank_dist['rank_2_5'] + rank_dist['rank_6_10']
    hit_20_total = hit_10_total + rank_dist['rank_11_20']
    hit_50_total = hit_20_total + rank_dist['rank_21_50']
    hit_100_total = hit_50_total + rank_dist['rank_51_100']

    print(f"\nTotal Analyzed: {total_r}")
    print(f"  Rank 1          : {rank_dist['rank_1']:>3} ({rank_dist['rank_1']/total_r*100:.1f}%)")
    print(f"  Rank 2-5        : {rank_dist['rank_2_5']:>3} ({rank_dist['rank_2_5']/total_r*100:.1f}%)")
    print(f"  Rank 6-10       : {rank_dist['rank_6_10']:>3} ({rank_dist['rank_6_10']/total_r*100:.1f}%)")
    print(f"  --> TOTAL HIT@10: {hit_10_total:>3} ({hit_10_total/total_r*100:.1f}%)")
    print(f"  Rank 11-20      : {rank_dist['rank_11_20']:>3} ({rank_dist['rank_11_20']/total_r*100:.1f}%)")
    print(f"  --> TOTAL HIT@20: {hit_20_total:>3} ({hit_20_total/total_r*100:.1f}%)")
    print(f"  Rank 21-50      : {rank_dist['rank_21_50']:>3} ({rank_dist['rank_21_50']/total_r*100:.1f}%)")
    print(f"  --> TOTAL HIT@50: {hit_50_total:>3} ({hit_50_total/total_r*100:.1f}%)")
    print(f"  Rank 51-100     : {rank_dist['rank_51_100']:>3} ({rank_dist['rank_51_100']/total_r*100:.1f}%)")
    print(f"  --> TOTAL HIT@100: {hit_100_total:>3} ({hit_100_total/total_r*100:.1f}%)")
    print(f"  Not in Top 100  : {rank_dist['not_in_top_100']:>3} ({rank_dist['not_in_top_100']/total_r*100:.1f}%)")

    print(f"\n--- PART 3: BREAKDOWN BY DIFFICULTY LEVEL ---")
    for diff in ["L1", "L2", "L3", "L4"]:
        d_total = sum(by_diff_ranks[diff].values())
        if d_total == 0: continue
        d_hit10 = by_diff_ranks[diff]["rank_1"] + by_diff_ranks[diff]["rank_2_5"] + by_diff_ranks[diff]["rank_6_10"]
        d_hit20 = d_hit10 + by_diff_ranks[diff]["rank_11_20"]
        d_hit50 = d_hit20 + by_diff_ranks[diff]["rank_21_50"]
        print(f"Level {diff:<3} ({d_total:>2} qs): Hit@10 = {d_hit10/d_total*100:>5.1f}% | Hit@20 = {d_hit20/d_total*100:>5.1f}% | Hit@50 = {d_hit50/d_total*100:>5.1f}% | Not in Top100: {by_diff_ranks[diff]['not_in_top_100']}")

    print(f"\n--- PART 4: SAMPLE MISS ANALYSIS (COMPETING CHUNKS INSPECTION) ---")
    competitor_has_facts_cnt = 0
    for m in miss_details:
        has_any_kw = any(c['has_target_keywords'] for c in m['top_competitors'])
        if has_any_kw:
            competitor_has_facts_cnt += 1

    print(f"Out of {len(miss_details)} missed queries, {competitor_has_facts_cnt} ({competitor_has_facts_cnt/len(miss_details)*100:.1f}%) have Top-3 competitors containing the TARGET KEYWORDS/FACTS!")

    for m in miss_details[:5]:
        print(f"\n[{m['qid']}] ({m['difficulty']} - {m['target_company']})")
        print(f"  Q: {m['question'][:90]}...")
        print(f"  GT ID: {m['gt_ids']} | Found at Rank: {m['found_rank']}")
        print(f"  Top Competitors ahead:")
        for comp in m['top_competitors']:
            print(f"    - Rank #{comp['rank']}: {comp['chunk_id']} (Score: {comp['score']}) Sec: '{comp['section']}' | Has Keywords? {comp['has_target_keywords']}")


if __name__ == "__main__":
    run_diagnostic()
