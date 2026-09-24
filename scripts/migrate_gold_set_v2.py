"""Script migration Ground Truth V2: Evidence-based Ground Truth Migration & Fact Signature.

Fixes Bug 1 (1A: Multi-occurrence fact candidates & 1B: TOC Contamination/Missing facts).
Applies Level-aware heuristic ranking, TOC filtering, and outputs comprehensive audit report.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_rag.config import (
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    GOLD_TEST_SET_FILE,
    get_collection_name,
)
from financial_rag.retrieval import DenseRetriever
from financial_rag.schema import extract_chunk_text
from financial_rag.testbed import (
    GoldQuestion,
    TICKER_MAP,
    load_gold_test_set,
    normalize_chunk_id,
)
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Stopwords to exclude from keyword distinctiveness matching
FINANCIAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at",
    "by", "from", "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "as", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "it", "its",
    "million", "billion", "dollar", "dollars", "total", "net", "company", "fiscal",
    "year", "ended", "period", "from", "compared", "percent", "reporting"
}


def normalize_financial_numbers(text: str) -> str:
    """Normalize numeric expressions, commas, accounting brackets, and unit scales."""
    text = str(text)
    # Remove thousand separators: 115,172 -> 115172
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    
    # Scale units: $93.736 billion -> 93736 million
    def scale_number(match):
        val = float(match.group(1))
        unit = match.group(2).lower()
        multipliers = {"billion": 1000, "million": 1, "thousand": 0.001}
        scaled = int(val * multipliers.get(unit, 1))
        return str(scaled)

    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*(billion|million|thousand)",
        scale_number,
        text,
        flags=re.IGNORECASE,
    )
    # Strip currency signs
    text = re.sub(r"[$€£¥]", "", text)
    # Accounting parentheses for negative numbers: (1,234) -> -1234
    text = re.sub(r"\((\d+)\)", r"-\1", text)
    return text


def extract_core_numbers(gold_q: GoldQuestion) -> List[str]:
    """Extract distinct financial numbers from financial_facts and gold_evidence keywords."""
    raw_nums = []
    # 1. From financial facts
    for fact in (gold_q.financial_facts or []):
        if isinstance(fact, dict) and fact.get("value") is not None:
            val = str(fact["value"]).replace(",", "").replace("$", "").strip()
            # Normalize float values like 391035.0 -> 391035
            if val.endswith(".0"):
                val = val[:-2]
            if val and val != "None":
                raw_nums.append(val)

    # 2. From evidence keywords
    for ev in (gold_q.gold_evidence or []):
        for kw in ev.get("keywords", []):
            cleaned = str(kw).replace(",", "").replace("$", "").replace("%", "").strip()
            # Catch numbers with digits
            found = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
            for f in found:
                if f.endswith(".0"):
                    f = f[:-2]
                raw_nums.append(f)

    # Filter out bare single digit numbers unless negative or part of specific ratios
    res = []
    for n in raw_nums:
        # Ignore single digits 0-9 without decimal/sign
        if len(n) == 1 and n.isdigit():
            continue
        if n not in res:
            res.append(n)
    return res


def extract_distinctive_keywords(gold_q: GoldQuestion) -> List[str]:
    """Extract keywords with stopwords and common financial tokens removed."""
    kws = []
    for ev in (gold_q.gold_evidence or []):
        for kw in ev.get("keywords", []):
            # Tokenize keyword phrase
            tokens = re.findall(r"[A-Za-z0-9]+", str(kw).lower())
            for t in tokens:
                if t not in FINANCIAL_STOPWORDS and not t.isdigit() and len(t) >= 3:
                    if t not in kws:
                        kws.append(t)
    return kws


def is_toc_chunk(chunk_payload: Dict[str, Any]) -> bool:
    """Detect if chunk is Table of Contents, Index, or Cover page."""
    sec = str(chunk_payload.get("section_title", "")).lower()
    ctype = str(chunk_payload.get("chunk_type", "")).lower()
    content = extract_chunk_text(chunk_payload).lower()[:300]
    
    if "table of contents" in sec or "table of contents" in content:
        return True
    if "index to consolidated" in sec or "index to consolidated" in content:
        return True
    if ctype == "toc":
        return True
    return False


@dataclass
class FactSignature:
    question_id: str
    ticker: str
    fiscal_year: Optional[int]
    core_numbers: List[str]
    keywords: List[str]
    difficulty: str
    min_keyword_ratio: float = 0.4


def build_signature(gold_q: GoldQuestion) -> FactSignature:
    ticker = getattr(gold_q, "target_company", "") or ""
    if not ticker and hasattr(gold_q, "document"):
        ticker = str(gold_q.document.get("company", ""))
    ticker = TICKER_MAP.get(ticker.lower(), ticker).upper()

    doc_year = None
    doc_info = getattr(gold_q, "document", {}) or {}
    if isinstance(doc_info, dict):
        p = str(doc_info.get("reporting_period", ""))
        m_yr = re.search(r"(20\d{2})", p)
        if m_yr:
            doc_year = int(m_yr.group(1))

    core_numbers = extract_core_numbers(gold_q)
    keywords = extract_distinctive_keywords(gold_q)

    return FactSignature(
        question_id=gold_q.id,
        ticker=ticker,
        fiscal_year=doc_year,
        core_numbers=core_numbers,
        keywords=keywords,
        difficulty=gold_q.difficulty,
        min_keyword_ratio=0.4,
    )


def chunk_satisfies_signature(chunk_payload: Dict[str, Any], sig: FactSignature) -> bool:
    """Check if chunk payload satisfies fact signature requirements."""
    # 0. Exclude TOC chunks
    if is_toc_chunk(chunk_payload):
        return False

    # 1. Ticker Check (if not MULTI)
    chunk_ticker = str(chunk_payload.get("ticker", "")).upper()
    if sig.ticker and sig.ticker != "MULTI" and chunk_ticker and chunk_ticker != sig.ticker:
        return False

    # 2. Fiscal Year Check (if chunk explicitly has year)
    chunk_year = chunk_payload.get("fiscal_year")
    if sig.fiscal_year and chunk_year and sig.ticker != "MULTI":
        try:
            if int(chunk_year) != sig.fiscal_year:
                return False
        except (ValueError, TypeError):
            pass

    # Extract normalized text
    text = extract_chunk_text(chunk_payload)
    text_norm = normalize_financial_numbers(text).lower()

    # 3. Number Check (100% of core numbers must be present)
    if sig.core_numbers:
        for n in sig.core_numbers:
            n_clean = n.lower()
            if n_clean not in text_norm:
                return False

    # 4. Keyword Check (ROUGE-1 Multiset Recall >= 0.4)
    if sig.keywords:
        matched_kw = sum(1 for kw in sig.keywords if kw.lower() in text_norm)
        if matched_kw / len(sig.keywords) < sig.min_keyword_ratio:
            return False

    return True


def rank_candidates_by_level(
    candidates: List[Dict[str, Any]],
    gold_q: GoldQuestion,
) -> List[str]:
    """Level-aware Heuristic Ranking for candidate chunks.
    
    L1, L2, L3: Prioritize Table (table_atomic) > Narrative
    L4: Prioritize Narrative (MD&A, Footnotes) > Table
    """
    level = gold_q.difficulty
    gold_sec = ""
    gold_pages = []
    for ev in (gold_q.gold_evidence or []):
        sec = str(ev.get("section", "")).lower()
        if sec and not gold_sec:
            gold_sec = sec
        pg = ev.get("page")
        if pg is not None:
            try:
                gold_pages.append(int(pg))
            except ValueError:
                pass

    scored_cands = []
    for cand in candidates:
        cid = cand.get("chunk_id", "")
        ctype = cand.get("chunk_type", "")
        sec_title = str(cand.get("section_title", "")).lower()
        pages = [int(p) for p in cand.get("source_pages", []) if str(p).isdigit()]

        score = 0.0

        # 1. Section Title Matching
        if gold_sec and (gold_sec in sec_title or sec_title in gold_sec):
            score += 10.0

        # 2. Level-Aware Chunk Type
        if level in ["L1", "L2", "L3"]:
            if ctype == "table_atomic":
                score += 5.0
            elif "table" in ctype:
                score += 3.0
            else:
                score += 1.0
        elif level == "L4":
            if "narrative" in ctype or "text" in ctype or ctype == "footnote":
                score += 5.0
            elif ctype == "table_atomic":
                score += 1.0

        # 3. Page Proximity
        if gold_pages and pages:
            min_dist = min(abs(cp - gp) for cp in pages for gp in gold_pages)
            score -= min_dist * 0.5

        scored_cands.append((score, cid))

    # Sort descending by score
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    return [cid for _, cid in scored_cands]


def migrate_gold_test_set():
    logger.info("=" * 80)
    logger.info("STARTING EVIDENCE-BASED GROUND TRUTH MIGRATION (V2)")
    logger.info("=" * 80)

    # 1. Create safety backup of dataset
    backup_file = GOLD_TEST_SET_FILE.with_suffix(".jsonl.bak")
    shutil.copyfile(GOLD_TEST_SET_FILE, backup_file)
    logger.info(f"Created safety backup at: {backup_file}")

    questions = load_gold_test_set(GOLD_TEST_SET_FILE)
    logger.info(f"Loaded {len(questions)} questions from {GOLD_TEST_SET_FILE}")

    retriever = DenseRetriever()
    methods_to_migrate = ["method2_deterministic", "method5_proposed_golden_hybrid"]

    migration_audit = {
        "total_questions": len(questions),
        "methods": {},
    }

    for method_key in methods_to_migrate:
        coll_name = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, method_key)
        logger.info(f"\nProcessing {method_key} (Collection: {coll_name})...")

        stats = {
            "chunking_failure": 0,
            "single_match": 0,
            "heuristic_ranked": 0,
            "annotation_error": 0,
            "abstention_skipped": 0,
        }

        for q in questions:
            # Skip L5 (unanswerable / abstention)
            if q.difficulty == "L5" or not q.answerable:
                stats["abstention_skipped"] += 1
                q.gold_chunk_ids[method_key] = {
                    "primary": [],
                    "secondary": [],
                    "chunking_coverage": "none",
                    "migration_reason": "abstention_no_ground_truth",
                }
                continue

            sig = build_signature(q)

            # Build Qdrant filter
            filters = []
            if sig.ticker and sig.ticker != "MULTI":
                filters.append(FieldCondition(key="ticker", match=MatchValue(value=sig.ticker)))
            if sig.fiscal_year and sig.ticker != "MULTI":
                filters.append(FieldCondition(key="fiscal_year", match=MatchValue(value=sig.fiscal_year)))

            all_pts = retriever.client.scroll(
                coll_name,
                scroll_filter=Filter(must=filters) if filters else None,
                limit=1000,
            )[0]

            # Find matching candidates
            matching_cands = []
            for pt in all_pts:
                if chunk_satisfies_signature(pt.payload, sig):
                    matching_cands.append(pt.payload)

            num_cands = len(matching_cands)

            if num_cands == 0:
                # 0 candidates -> Chunking Coverage Failure (Method lost evidence)
                stats["chunking_failure"] += 1
                q.gold_chunk_ids[method_key] = {
                    "primary": [],
                    "secondary": [],
                    "chunking_coverage": "none",
                    "migration_reason": "chunking_lost_evidence",
                }
            elif num_cands == 1:
                # 1 candidate -> Single unambiguous match
                stats["single_match"] += 1
                cid = matching_cands[0].get("chunk_id")
                q.gold_chunk_ids[method_key] = {
                    "primary": [cid],
                    "secondary": [],
                    "chunking_coverage": "full",
                    "migration_reason": "single_unambiguous_match",
                }
            elif 2 <= num_cands <= 5:
                # 2-5 candidates -> Level-aware Heuristic Ranking
                stats["heuristic_ranked"] += 1
                ranked_ids = rank_candidates_by_level(matching_cands, q)
                q.gold_chunk_ids[method_key] = {
                    "primary": [ranked_ids[0]],
                    "secondary": ranked_ids[1:],
                    "chunking_coverage": "full",
                    "migration_reason": "heuristic_ranked_match",
                }
            else:
                # > 5 candidates -> Flag for annotation ambiguity
                stats["annotation_error"] += 1
                ranked_ids = rank_candidates_by_level(matching_cands, q)
                q.gold_chunk_ids[method_key] = {
                    "primary": [ranked_ids[0]],
                    "secondary": ranked_ids[1:5],
                    "chunking_coverage": "full",
                    "migration_reason": "annotation_ambiguity_high_candidate_count",
                    "candidate_count": num_cands,
                }

        logger.info(f"Migration summary for {method_key}: {stats}")
        migration_audit["methods"][method_key] = stats

    # 2. Sanity Check Quality Gate
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SANITY CHECK QUALITY GATE")
    logger.info("=" * 80)
    
    sanity_failures = []
    total_checked = 0

    for q in questions:
        if q.difficulty == "L5":
            continue
        sig = build_signature(q)
        for m in methods_to_migrate:
            m_data = q.gold_chunk_ids.get(m, {})
            if isinstance(m_data, dict):
                cov = m_data.get("chunking_coverage")
                if cov == "none":
                    continue
                primary_ids = m_data.get("primary", [])
                if not primary_ids:
                    sanity_failures.append((q.id, m, "Missing primary ID"))
                    continue
                
                target_id = primary_ids[0]
                coll_m = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, m)
                pts = retriever.client.scroll(
                    coll_m,
                    scroll_filter=Filter(must=[FieldCondition(key="chunk_id", match=MatchValue(value=target_id))]),
                    limit=1,
                )[0]
                total_checked += 1
                if not pts or not chunk_satisfies_signature(pts[0].payload, sig):
                    sanity_failures.append((q.id, m, target_id))

    fail_rate = len(sanity_failures) / total_checked if total_checked > 0 else 0.0
    logger.info(f"Sanity Check: {total_checked - len(sanity_failures)}/{total_checked} passed ({1.0 - fail_rate:.1%})")
    
    if sanity_failures:
        logger.warning(f"Sanity failures encountered: {len(sanity_failures)} items (rate: {fail_rate:.2%})")
        for f in sanity_failures[:5]:
            logger.warning(f"  Fail item: {f}")

    migration_audit["sanity_check"] = {
        "total_checked": total_checked,
        "failures_count": len(sanity_failures),
        "failure_rate": fail_rate,
        "status": "PASS" if fail_rate <= 0.05 else "FAIL",
    }

    # 3. Decision on Overwriting
    audit_file = ROOT / "outputs" / "retrieval" / "v0_dense_baseline" / "migration_audit_report.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(migration_audit, f, indent=2, ensure_ascii=False)
    logger.info(f"Audit report saved to: {audit_file}")

    if fail_rate > 0.05:
        logger.error(f"HARD FAIL: Sanity check failure rate {fail_rate:.1%} exceeds 5% threshold! Aborting overwrite.")
        sys.exit(1)
    else:
        # Write migrated questions to v0_gold_questions.jsonl
        logger.info(f"SOFT FAIL / PASS: Sanity check passed quality gate (rate <= 5%). Writing to {GOLD_TEST_SET_FILE}...")
        with open(GOLD_TEST_SET_FILE, "w", encoding="utf-8") as f:
            for q in questions:
                # Convert dataclass to dict
                data = asdict(q)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info("Successfully updated gold test set with Evidence-based Ground Truth!")

if __name__ == "__main__":
    migrate_gold_test_set()
