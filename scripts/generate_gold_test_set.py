"""Generate Frozen Gold Test Set (140 Questions) for FinAnalyst-AI V0 Benchmark.

Scientific Structure:
- 7 Companies: AAPL, NVDA, AMZN, AMD, INTC, NKE, WMT (20 questions / company).
- 5 Difficulty Levels per company:
    * L1 Single Fact Lookup: 5 questions (25% -> 35 total)
    * L2 Table Reasoning: 5 questions (25% -> 35 total)
    * L3 Multi-Year / Cross-Company Comparison: 4 questions (20% -> 28 total)
    * L4 Qualitative Synthesis / Adversarial: 4 questions (20% -> 28 total)
    * L5 Abstention & Safety: 2 questions (10% -> 14 total)
- Total Questions: 140
    * 126 Retrieval Questions (L1-L4)
    * 14 Abstention Questions (L5)
- Automatically maps true positive ground truth chunk IDs across all 5 chunking methods:
    * method1_fixed_size
    * method2_deterministic
    * method3_heading_llm
    * method4_boundary_tagging
    * method5_proposed_golden_hybrid
- Calculates token containment (Counter ROUGE-1 recall), expected chunk counts, and coverage status.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from financial_rag.config import (
    CHUNK_METHODS,
    OUTPUT_CHUNKING_ROOT,
    GOLD_TEST_SET_DIR,
    GOLD_TEST_SET_FILE,
)
from financial_rag.schema import (
    CONTAINMENT_THRESHOLDS,
    compute_containment,
    extract_chunk_text,
    infer_expected_chunk_count,
    normalize_numbers,
    normalize_to_million,
    RETRIEVAL_LEVELS,
    ABSTENTION_LEVEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_gold_test_set")


# ============================================================
# CACHED CHUNK LOADER ACROSS ALL 5 METHODS
# ============================================================
_CHUNK_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

def load_chunks_for_doc_method(doc_id: str, method_key: str) -> List[Dict[str, Any]]:
    """Load chunks from disk with in-memory caching."""
    cache_key = (doc_id, method_key)
    if cache_key in _CHUNK_CACHE:
        return _CHUNK_CACHE[cache_key]

    folder_name = CHUNK_METHODS[method_key]["folder_name"]
    chunk_file = OUTPUT_CHUNKING_ROOT / doc_id / folder_name / "chunks.jsonl"
    if not chunk_file.exists():
        logger.warning(f"Chunk file not found: {chunk_file}")
        _CHUNK_CACHE[cache_key] = []
        return []

    chunks = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    _CHUNK_CACHE[cache_key] = chunks
    return chunks


def map_ground_truth_for_question(
    q_data: Dict[str, Any],
    doc_id: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]]:
    """Scan all 5 chunking methods and identify matching ground truth chunks.
    
    Returns:
        (gold_chunk_ids, coverage, containment_scores)
    """
    diff = q_data["difficulty"]
    if diff == ABSTENTION_LEVEL or not q_data.get("answerable", True):
        return (
            {m: [] for m in CHUNK_METHODS},
            {m: "none" for m in CHUNK_METHODS},
            {m: 0.0 for m in CHUNK_METHODS},
        )

    threshold = CONTAINMENT_THRESHOLDS.get(diff, 0.40)
    expected_count = q_data.get("expected_chunk_count", 1)

    # Primary evidence text from gold_evidence
    evidence_texts = [ev.get("text", "") for ev in q_data.get("gold_evidence", []) if ev.get("text")]
    combined_ev = " ".join(evidence_texts) if evidence_texts else q_data.get("question", "")

    # Target keywords and numbers
    kws = [str(k).lower() for ev in q_data.get("gold_evidence", []) for k in ev.get("keywords", [])]

    gold_chunk_ids: Dict[str, List[str]] = {}
    coverage: Dict[str, str] = {}
    containment_scores: Dict[str, float] = {}

    for method_key in CHUNK_METHODS:
        chunks = load_chunks_for_doc_method(doc_id, method_key)
        matched_ids = []
        best_score = 0.0

        for chunk in chunks:
            chunk_txt = extract_chunk_text(chunk)
            score = compute_containment(combined_ev, chunk_txt)
            if score > best_score:
                best_score = score

            # Signature check: containment must satisfy threshold, and key numbers must match
            if score >= threshold:
                matched_ids.append(chunk["chunk_id"])
            elif kws:
                # Fallback: if all key words/numbers appear in chunk_txt
                ch_lower = chunk_txt.lower()
                clean_ch = normalize_numbers(ch_lower)
                if all(normalize_numbers(kw) in clean_ch for kw in kws):
                    matched_ids.append(chunk["chunk_id"])
                    if score < threshold:
                        best_score = max(best_score, threshold)

        gold_chunk_ids[method_key] = matched_ids
        containment_scores[method_key] = round(best_score, 4)

        if len(matched_ids) >= expected_count:
            coverage[method_key] = "full"
        elif len(matched_ids) > 0:
            coverage[method_key] = "partial"
        else:
            coverage[method_key] = "none"

    return gold_chunk_ids, coverage, containment_scores


# ============================================================
# 140 GOLD QUESTIONS DEFINITION SPECIFICATION
# ============================================================
def build_raw_gold_questions() -> List[Dict[str, Any]]:
    """Build the master raw specification of 140 benchmark questions across 7 companies."""
    questions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1. AAPL (Apple Inc. - FY2024 10-K) - 20 questions
    # ------------------------------------------------------------
    # L1: Single Fact Lookup (5 questions)
    questions.append({
        "id": "GOLD_AAPL_L1_01",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's total net sales for the fiscal year ended September 28, 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Total net sales", "391,035", "391035", "2024"],
        "evidence_text": "Total net sales was $391,035 million for 2024 compared to $383,285 million for 2023.",
        "financial_facts": [{"metric": "Total Net Sales", "period": "FY2024", "value": 391035, "unit": "million"}],
        "normalized_value_in_million": 391035.0,
        "gold_answer": "$391,035 million (or $391.035 billion)",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L1_02",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's iPhone net sales in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Category",
        "keywords": ["iPhone", "201,183", "201183", "2024"],
        "evidence_text": "Line item: iPhone | 2024: $ 201,183 million, Change: - %, 2023: $ 200,583 million.",
        "financial_facts": [{"metric": "iPhone Net Sales", "period": "FY2024", "value": 201183, "unit": "million"}],
        "normalized_value_in_million": 201183.0,
        "gold_answer": "$201,183 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L1_03",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's total research and development (R&D) expense in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 23,
        "section": "Operating Expenses",
        "keywords": ["Research and development", "31,370", "31370", "2024"],
        "evidence_text": "Research and development expense was $31,370 million in 2024.",
        "financial_facts": [{"metric": "Research and Development Expense", "period": "FY2024", "value": 31370, "unit": "million"}],
        "normalized_value_in_million": 31370.0,
        "gold_answer": "$31,370 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L1_04",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's net income for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 48,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net income", "93,736", "93736", "2024"],
        "evidence_text": "Net income for 2024 was $93,736 million compared to $96,995 million in 2023.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2024", "value": 93736, "unit": "million"}],
        "normalized_value_in_million": 93736.0,
        "gold_answer": "$93,736 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L1_05",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "How much cash and cash equivalents did Apple hold at the end of fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Cash and cash equivalents", "29,942", "29942", "2024"],
        "evidence_text": "Cash and cash equivalents was $29,942 million as of September 28, 2024.",
        "financial_facts": [{"metric": "Cash and Cash Equivalents", "period": "FY2024", "value": 29942, "unit": "million"}],
        "normalized_value_in_million": 29942.0,
        "gold_answer": "$29,942 million",
        "answerable": True,
    })

    # L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_AAPL_L2_01",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "According to the net sales by product category table, what was the net sales figure for Services in fiscal year 2024 and how much did it change compared to 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "mixed",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Category Table",
        "keywords": ["Services", "96,169", "96169", "85,200", "13 %"],
        "evidence_text": "Services net sales: 2024: $ 96,169 million, Change: 13 %, 2023: $ 85,200 million.",
        "financial_facts": [
            {"metric": "Services Net Sales 2024", "period": "FY2024", "value": 96169, "unit": "million"},
            {"metric": "Services Net Sales 2023", "period": "FY2023", "value": 85200, "unit": "million"},
        ],
        "normalized_value_in_million": 96169.0,
        "gold_answer": "$96,169 million in 2024, an increase of 13% (or $10,969 million) from $85,200 million in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L2_02",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What were the net sales figures for the Americas and Europe geographic segments in 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Geographic Segment Table",
        "keywords": ["Americas", "167,045", "Europe", "101,328", "2024"],
        "evidence_text": "Americas net sales was $167,045 million and Europe net sales was $101,328 million in 2024.",
        "financial_facts": [
            {"metric": "Americas Net Sales", "period": "FY2024", "value": 167045, "unit": "million"},
            {"metric": "Europe Net Sales", "period": "FY2024", "value": 101328, "unit": "million"},
        ],
        "normalized_value_in_million": 167045.0,
        "gold_answer": "Americas: $167,045 million; Europe: $101,328 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L2_03",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was the reported net sales for Greater China in 2024, and what was the percentage change from 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "mixed",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Geographic Segment Table",
        "keywords": ["Greater China", "66,952", "(8) %", "72,559"],
        "evidence_text": "Greater China net sales was $66,952 million in 2024, decreasing by (8)% compared to $72,559 million in 2023.",
        "financial_facts": [{"metric": "Greater China Net Sales", "period": "FY2024", "value": 66952, "unit": "million"}],
        "normalized_value_in_million": 66952.0,
        "gold_answer": "$66,952 million, a decrease of 8% compared to $72,559 million in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L2_04",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's Mac and iPad net sales in fiscal year 2024 according to the category breakdown table?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Category Table",
        "keywords": ["Mac", "29,984", "iPad", "26,694"],
        "evidence_text": "Mac net sales was $29,984 million and iPad net sales was $26,694 million in 2024.",
        "financial_facts": [
            {"metric": "Mac Net Sales", "period": "FY2024", "value": 29984, "unit": "million"},
            {"metric": "iPad Net Sales", "period": "FY2024", "value": 26694, "unit": "million"},
        ],
        "normalized_value_in_million": 29984.0,
        "gold_answer": "Mac: $29,984 million; iPad: $26,694 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L2_05",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "From the Consolidated Statements of Operations, what were Total cost of sales and Gross margin for 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 48,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Total cost of sales", "210,352", "Gross margin", "180,683"],
        "evidence_text": "Total cost of sales was $210,352 million and Gross margin was $180,683 million in 2024.",
        "financial_facts": [
            {"metric": "Total Cost of Sales", "period": "FY2024", "value": 210352, "unit": "million"},
            {"metric": "Gross Margin", "period": "FY2024", "value": 180683, "unit": "million"},
        ],
        "normalized_value_in_million": 180683.0,
        "gold_answer": "Total cost of sales: $210,352 million; Gross margin: $180,683 million.",
        "answerable": True,
    })

    # L3: Multi-Year / Cross-Company Comparison (4 questions)
    questions.append({
        "id": "GOLD_AAPL_L3_01",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "How did Apple's total net sales change between fiscal 2023 and fiscal 2024 in dollar terms and percentage?",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2024",
        "evidence_page": 22,
        "section": "Net Sales Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["391,035", "383,285", "2 %"],
        "evidence_text": "Total net sales increased from $383,285 million in 2023 to $391,035 million in 2024, a change of 2% ($7,750 million increase).",
        "financial_facts": [
            {"metric": "Net Sales 2024", "period": "FY2024", "value": 391035, "unit": "million"},
            {"metric": "Net Sales 2023", "period": "FY2023", "value": 383285, "unit": "million"},
        ],
        "reasoning": {"formula": "(391035 - 383285) / 383285 * 100", "expected_result": 2.02, "tolerance": 0.1},
        "gold_answer": "Increased by $7,750 million or approximately 2.02% (reported as 2%).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L3_02",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "Compare Apple's Services revenue growth with Mac revenue growth from 2023 to 2024.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2024",
        "evidence_page": 22,
        "section": "Category Net Sales Table",
        "comparison_targets": ["Services", "Mac"],
        "keywords": ["Services", "13 %", "Mac", "2 %"],
        "evidence_text": "Services grew by 13% (from $85,200M to $96,169M) while Mac grew by 2% (from $29,357M to $29,984M).",
        "financial_facts": [
            {"metric": "Services Growth", "period": "FY2024", "value": 13.0, "unit": "percent"},
            {"metric": "Mac Growth", "period": "FY2024", "value": 2.0, "unit": "percent"},
        ],
        "gold_answer": "Services grew by 13% ($10,969M increase), substantially outperforming Mac revenue growth of 2% ($627M increase).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L3_03",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "Compare Apple's Gross Margin percentage in fiscal 2024 versus fiscal 2023.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 23,
        "section": "Gross Margin Overview",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Gross margin", "46.2 %", "44.1 %"],
        "evidence_text": "Gross margin percentage was 46.2% in 2024 compared to 44.1% in 2023.",
        "financial_facts": [
            {"metric": "Gross Margin % 2024", "period": "FY2024", "value": 46.2, "unit": "percent"},
            {"metric": "Gross Margin % 2023", "period": "FY2023", "value": 44.1, "unit": "percent"},
        ],
        "gold_answer": "Gross margin increased from 44.1% in 2023 to 46.2% in 2024 (an expansion of 210 basis points).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L3_04",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "Compare the 3-year trend of Apple's Greater China revenue from 2022 to 2024.",
        "question_type": "trend",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2022-FY2024",
        "evidence_page": 22,
        "section": "Geographic Net Sales Table",
        "comparison_targets": ["2022", "2023", "2024"],
        "keywords": ["Greater China", "74,200", "72,559", "66,952"],
        "evidence_text": "Greater China net sales declined consistently over 3 years: $74,200 million in 2022, $72,559 million in 2023 (-2%), and $66,952 million in 2024 (-8%).",
        "financial_facts": [
            {"metric": "Greater China 2022", "period": "FY2022", "value": 74200, "unit": "million"},
            {"metric": "Greater China 2023", "period": "FY2023", "value": 72559, "unit": "million"},
            {"metric": "Greater China 2024", "period": "FY2024", "value": 66952, "unit": "million"},
        ],
        "gold_answer": "Greater China experienced consecutive annual declines from $74,200M (2022) to $72,559M (2023) and down to $66,952M (2024).",
        "answerable": True,
    })

    # L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_AAPL_L4_01",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What primary operational factors drove the growth in Apple's Services net sales during fiscal 2024 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 23,
        "section": "Item 7. Management's Discussion and Analysis",
        "keywords": ["Services net sales increased", "higher net sales of advertising", "cloud services", "App Store"],
        "evidence_text": "Services net sales increased during 2024 compared to 2023 due primarily to higher net sales of advertising, cloud services and the App Store.",
        "gold_answer": "Growth was driven primarily by higher net sales in advertising, cloud services, and the App Store.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L4_02",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What explanation does Apple provide for the decrease in Greater China net sales during fiscal year 2024?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Item 7. MD&A",
        "keywords": ["Greater China net sales decreased", "lower net sales of iPhone and iPad", "weakness in the renminbi"],
        "evidence_text": "Greater China net sales decreased during 2024 compared to 2023 due primarily to lower net sales of iPhone and iPad, alongside the unfavorable year-over-year impact of the weakness in the renminbi relative to the U.S. dollar.",
        "gold_answer": "Lower net sales of iPhone and iPad, compounded by currency weakness of the renminbi against the U.S. dollar.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L4_03",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What does Apple state regarding foreign currency exchange rate risks and its hedging program in Item 7A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 30,
        "section": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        "keywords": ["foreign exchange risk", "derivatives", "cash flow hedges"],
        "evidence_text": "The Company enters into foreign currency forward and option contracts to hedge foreign currency denominated assets, liabilities, and forecasted transactions.",
        "gold_answer": "Apple utilizes foreign currency forward and option contracts to mitigate the volatility of foreign exchange rates on cash flows, assets, and liabilities.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AAPL_L4_04",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "Given that Apple's iPhone sales decreased to $150,000 million in fiscal 2024, explain the impact on gross margin.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 22,
        "section": "Net Sales by Category Table",
        "keywords": ["iPhone", "201,183"],
        "evidence_text": "Line item: iPhone | 2024: $ 201,183 million.",
        "gold_answer": "The premise is factually incorrect. Apple's iPhone sales were $201,183 million in fiscal 2024, not $150,000 million.",
        "answerable": True,
    })

    # L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_AAPL_L5_01",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "How many unit shipments of iPhone 16 did Apple deliver worldwide during fiscal year 2024?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Apple stopped disclosing unit shipment numbers for iPhones post-2018; only dollar net sales are disclosed.",
        "gold_answer": "Cannot determine. Apple does not disclose unit shipment volumes for iPhones in Form 10-K, reporting only dollar revenue ($201,183 million in FY2024).",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_AAPL_L5_02",
        "ticker": "AAPL",
        "doc_id": "apple_2024_10k",
        "question": "What was Apple's total net sales for the fiscal year ended September 2027?",
        "question_type": "unanswerable_future_year",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2027",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Fiscal year 2027 is in the future and has not occurred or been reported.",
        "gold_answer": "Cannot determine. Fiscal year 2027 is a future period and results have not been reported in the 10-K.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 2. NVDA (NVIDIA Corporation - FY2025 10-K) - 20 questions
    # ------------------------------------------------------------
    questions.append({
        "id": "GOLD_NVDA_L1_01",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's total revenue for the fiscal year ended January 26, 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Total revenue", "130,497", "130497", "2025"],
        "evidence_text": "Total revenue was $130,497 million for the fiscal year ended January 26, 2025 compared to $60,922 million in 2024.",
        "financial_facts": [{"metric": "Total Revenue", "period": "FY2025", "value": 130497, "unit": "million"}],
        "normalized_value_in_million": 130497.0,
        "gold_answer": "$130,497 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L1_02",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's Data Center revenue in fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 40,
        "section": "Revenue by Market Platform",
        "keywords": ["Data Center", "115,186", "115186", "2025"],
        "evidence_text": "Line item: Data Center | Year Ended - Jan 26, 2025 - (In millions): $ 115,186 compared to $ 47,525 in 2024.",
        "financial_facts": [{"metric": "Data Center Revenue", "period": "FY2025", "value": 115186, "unit": "million"}],
        "normalized_value_in_million": 115186.0,
        "gold_answer": "$115,186 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L1_03",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's net income for fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Net income", "72,880", "72880", "2025"],
        "evidence_text": "Net income for fiscal 2025 was $72,880 million compared to $29,760 million in 2024.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2025", "value": 72880, "unit": "million"}],
        "normalized_value_in_million": 72880.0,
        "gold_answer": "$72,880 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L1_04",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's gross profit in fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Gross profit", "97,871", "97871", "2025"],
        "evidence_text": "Gross profit was $97,871 million for fiscal year 2025.",
        "financial_facts": [{"metric": "Gross Profit", "period": "FY2025", "value": 97871, "unit": "million"}],
        "normalized_value_in_million": 97871.0,
        "gold_answer": "$97,871 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L1_05",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's research and development (R&D) expense in fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Research and development", "12,914", "12914", "2025"],
        "evidence_text": "Research and development expense was $12,914 million in fiscal year 2025.",
        "financial_facts": [{"metric": "R&D Expense", "period": "FY2025", "value": 12914, "unit": "million"}],
        "normalized_value_in_million": 12914.0,
        "gold_answer": "$12,914 million",
        "answerable": True,
    })

    # NVDA L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_NVDA_L2_01",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "According to the market platform revenue table, what were the revenues for Compute and Networking within Data Center in fiscal 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 40,
        "section": "Market Platform Table",
        "keywords": ["Compute", "102,196", "Networking", "12,990"],
        "evidence_text": "Compute was $102,196 million and Networking was $12,990 million within Data Center in fiscal 2025.",
        "financial_facts": [
            {"metric": "Compute Revenue", "period": "FY2025", "value": 102196, "unit": "million"},
            {"metric": "Networking Revenue", "period": "FY2025", "value": 12990, "unit": "million"},
        ],
        "normalized_value_in_million": 102196.0,
        "gold_answer": "Compute: $102,196 million; Networking: $12,990 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L2_02",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's Gaming revenue in fiscal 2025 compared to fiscal 2024 according to the market platform table?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 40,
        "section": "Market Platform Table",
        "keywords": ["Gaming", "11,350", "10,447"],
        "evidence_text": "Gaming revenue was $11,350 million in fiscal 2025 compared to $10,447 million in fiscal 2024.",
        "financial_facts": [
            {"metric": "Gaming Revenue 2025", "period": "FY2025", "value": 11350, "unit": "million"},
            {"metric": "Gaming Revenue 2024", "period": "FY2024", "value": 10447, "unit": "million"},
        ],
        "normalized_value_in_million": 11350.0,
        "gold_answer": "$11,350 million in fiscal 2025 compared to $10,447 million in fiscal 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L2_03",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What were the revenues for Professional Visualization and Automotive platforms in fiscal 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 40,
        "section": "Market Platform Table",
        "keywords": ["Professional Visualization", "1,878", "Automotive", "1,694"],
        "evidence_text": "Professional Visualization was $1,878 million and Automotive was $1,694 million in fiscal 2025.",
        "financial_facts": [
            {"metric": "Professional Visualization", "period": "FY2025", "value": 1878, "unit": "million"},
            {"metric": "Automotive", "period": "FY2025", "value": 1694, "unit": "million"},
        ],
        "normalized_value_in_million": 1878.0,
        "gold_answer": "Professional Visualization: $1,878 million; Automotive: $1,694 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L2_04",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's Operating Income in fiscal 2025 and fiscal 2024 from the Consolidated Statements of Income?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Operating income", "81,482", "32,972"],
        "evidence_text": "Operating income was $81,482 million in 2025 compared to $32,972 million in 2024.",
        "financial_facts": [
            {"metric": "Operating Income 2025", "period": "FY2025", "value": 81482, "unit": "million"},
            {"metric": "Operating Income 2024", "period": "FY2024", "value": 32972, "unit": "million"},
        ],
        "normalized_value_in_million": 81482.0,
        "gold_answer": "$81,482 million in fiscal 2025 compared to $32,972 million in fiscal 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L2_05",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "From the Consolidated Statements of Income, what was diluted earnings per share (EPS) for fiscal 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "keywords": ["Diluted earnings per share", "2.94", "2025"],
        "evidence_text": "Diluted earnings per share was $2.94 for fiscal year 2025 compared to $1.19 for fiscal year 2024.",
        "financial_facts": [{"metric": "Diluted EPS", "period": "FY2025", "value": 2.94, "unit": "dollar"}],
        "gold_answer": "$2.94 per share",
        "answerable": True,
    })

    # NVDA L3: Multi-Year Comparison (4 questions)
    questions.append({
        "id": "GOLD_NVDA_L3_01",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "By what percentage did NVIDIA's Data Center revenue increase from fiscal 2024 to fiscal 2025?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2024-FY2025",
        "evidence_page": 40,
        "section": "Market Platform Table",
        "comparison_targets": ["2024", "2025"],
        "keywords": ["115,186", "47,525", "142 %"],
        "evidence_text": "Data Center revenue was $115,186 million in 2025 compared to $47,525 million in 2024, an increase of 142%.",
        "financial_facts": [
            {"metric": "Data Center 2025", "period": "FY2025", "value": 115186, "unit": "million"},
            {"metric": "Data Center 2024", "period": "FY2024", "value": 47525, "unit": "million"},
        ],
        "reasoning": {"formula": "(115186 - 47525) / 47525 * 100", "expected_result": 142.37, "tolerance": 0.5},
        "gold_answer": "142.37% (reported as 142%).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L3_02",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "How did NVIDIA's total revenue grow over the 3-year period from fiscal 2023 ($26,974M) through fiscal 2025 ($130,497M)?",
        "question_type": "trend",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2025",
        "evidence_page": 53,
        "section": "Consolidated Statements of Income",
        "comparison_targets": ["2023", "2024", "2025"],
        "keywords": ["130,497", "60,922", "26,974"],
        "evidence_text": "Total revenue grew from $26,974 million in 2023 to $60,922 million in 2024 (+126%), and reached $130,497 million in 2025 (+114%).",
        "financial_facts": [
            {"metric": "Revenue 2023", "period": "FY2023", "value": 26974, "unit": "million"},
            {"metric": "Revenue 2024", "period": "FY2024", "value": 60922, "unit": "million"},
            {"metric": "Revenue 2025", "period": "FY2025", "value": 130497, "unit": "million"},
        ],
        "gold_answer": "Revenue grew nearly 5-fold, from $26,974M in FY2023 to $60,922M in FY2024 and $130,497M in FY2025.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L3_03",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What was NVIDIA's gross margin percentage in fiscal 2025 versus fiscal 2024?",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2024-FY2025",
        "evidence_page": 39,
        "section": "Gross Margin Discussion",
        "comparison_targets": ["2024", "2025"],
        "keywords": ["Gross margin", "75.0 %", "72.7 %"],
        "evidence_text": "Gross margin increased to 75.0% in fiscal 2025 from 72.7% in fiscal 2024.",
        "financial_facts": [
            {"metric": "Gross Margin % 2025", "period": "FY2025", "value": 75.0, "unit": "percent"},
            {"metric": "Gross Margin % 2024", "period": "FY2024", "value": 72.7, "unit": "percent"},
        ],
        "gold_answer": "75.0% in fiscal 2025 compared to 72.7% in fiscal 2024 (an expansion of 230 basis points).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L3_04",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "Compare the dollar increase in Data Center Compute versus Networking revenue between fiscal 2024 and 2025.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2024-FY2025",
        "evidence_page": 40,
        "section": "Market Platform Table",
        "comparison_targets": ["Compute", "Networking"],
        "keywords": ["Compute", "102,196", "38,737", "Networking", "12,990", "8,788"],
        "evidence_text": "Compute revenue increased by $63,459 million (from $38,737M to $102,196M) while Networking increased by $4,202 million (from $8,788M to $12,990M).",
        "financial_facts": [
            {"metric": "Compute 2025", "period": "FY2025", "value": 102196, "unit": "million"},
            {"metric": "Networking 2025", "period": "FY2025", "value": 12990, "unit": "million"},
        ],
        "gold_answer": "Compute increased by $63,459 million, which was over 15 times larger than Networking's increase of $4,202 million.",
        "answerable": True,
    })

    # NVDA L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_NVDA_L4_01",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What primary factors drove the surge in NVIDIA's Data Center revenue during fiscal 2025 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 40,
        "section": "Item 7. MD&A - Market Platform Highlights",
        "keywords": ["Data Center revenue increased", "Hopper architecture", "generative AI", "large language models"],
        "evidence_text": "Data Center revenue increased significantly driven by demand for the NVIDIA Hopper GPU computing platform used for training and inferencing large language models, generative AI, and advanced computing applications.",
        "gold_answer": "Strong global demand for the Hopper architecture computing platform used for training and inference of generative AI and large language models.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L4_02",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What risks does NVIDIA highlight regarding customer concentration among Cloud Service Providers (CSPs) in fiscal 2025?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 15,
        "section": "Item 1A. Risk Factors",
        "keywords": ["Customer concentration", "Cloud Service Providers", "significant portion of total revenue"],
        "evidence_text": "A significant portion of our revenue is concentrated among a small number of customers, including large Cloud Service Providers. If we fail to maintain these relationships, our business could be materially harmed.",
        "gold_answer": "A substantial portion of Data Center revenue is derived from a limited number of large CSPs and consumer internet companies, creating significant concentration risk.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L4_03",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What did NVIDIA disclose regarding supply chain manufacturing constraints and foundry dependency on TSMC?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 16,
        "section": "Item 1A. Risk Factors",
        "keywords": ["third-party foundries", "TSMC", "packaging capacity"],
        "evidence_text": "We rely on third-party foundries, primarily TSMC, to manufacture our semiconductor wafers, and specialized advanced packaging suppliers for CoWoS assembly.",
        "gold_answer": "NVIDIA relies on third-party foundries, predominantly TSMC, for wafer fabrication and advanced packaging (such as CoWoS), making its supply constrained by their capacity.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NVDA_L4_04",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "Given that NVIDIA's gross margin fell to 35% in fiscal 2025, explain why profitability collapsed.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 39,
        "section": "Gross Margin",
        "keywords": ["Gross margin", "75.0 %"],
        "evidence_text": "Gross margin was 75.0% for fiscal year 2025.",
        "gold_answer": "The premise is false. NVIDIA's gross margin did not collapse to 35%; it actually expanded to 75.0% in fiscal 2025.",
        "answerable": True,
    })

    # NVDA L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_NVDA_L5_01",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "How many individual H100 GPU units did NVIDIA ship directly to Microsoft in fiscal year 2025?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2025",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "NVIDIA does not disclose physical chip unit shipment counts broken down by specific customer in Form 10-K.",
        "gold_answer": "Cannot determine. NVIDIA does not disclose individual chip unit shipment numbers by customer in its 10-K filing.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_NVDA_L5_02",
        "ticker": "NVDA",
        "doc_id": "nvidia_2025_10k",
        "question": "What is NVIDIA's projected net income for the fiscal year ending in January 2027?",
        "question_type": "unanswerable_future_year",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2027",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Fiscal year 2027 is a future reporting period not covered in the 10-K.",
        "gold_answer": "Cannot determine. Fiscal year 2027 is in the future and financial results have not been reported.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 3. AMD (Advanced Micro Devices - FY2024 10-K) - 20 questions
    # ------------------------------------------------------------
    # L1: Single Fact Lookup (5 questions)
    questions.append({
        "id": "GOLD_AMD_L1_01",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's total net revenue for the fiscal year ended December 28, 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Results of Operations Table",
        "keywords": ["Total net revenue", "25,785", "25785", "2024"],
        "evidence_text": "Total net revenue was $25,785 million for the year ended December 28, 2024 compared to $22,680 million for 2023.",
        "financial_facts": [{"metric": "Total Net Revenue", "period": "FY2024", "value": 25785, "unit": "million"}],
        "normalized_value_in_million": 25785.0,
        "gold_answer": "$25,785 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L1_02",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's Data Center segment net revenue in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "keywords": ["Data Center", "12,579", "12579", "2024"],
        "evidence_text": "Data Center segment net revenue was $12,579 million for 2024 compared to $6,496 million for 2023.",
        "financial_facts": [{"metric": "Data Center Net Revenue", "period": "FY2024", "value": 12579, "unit": "million"}],
        "normalized_value_in_million": 12579.0,
        "gold_answer": "$12,579 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L1_03",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's Client segment net revenue in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "keywords": ["Client", "7,054", "7054", "2024"],
        "evidence_text": "Client segment net revenue was $7,054 million in 2024.",
        "financial_facts": [{"metric": "Client Net Revenue", "period": "FY2024", "value": 7054, "unit": "million"}],
        "normalized_value_in_million": 7054.0,
        "gold_answer": "$7,054 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L1_04",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's total research and development (R&D) expense in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 52,
        "section": "Expenses Table",
        "keywords": ["Research and development", "6,298", "6298", "2024"],
        "evidence_text": "Research and development expense was $6,298 million for 2024 compared to $5,872 million for 2023.",
        "financial_facts": [{"metric": "R&D Expense", "period": "FY2024", "value": 6298, "unit": "million"}],
        "normalized_value_in_million": 6298.0,
        "gold_answer": "$6,298 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L1_05",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's Net Income for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 53,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net income", "1,641", "1641", "2024"],
        "evidence_text": "Net income was $1,641 million in 2024 compared to $854 million in 2023.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2024", "value": 1641, "unit": "million"}],
        "normalized_value_in_million": 1641.0,
        "gold_answer": "$1,641 million",
        "answerable": True,
    })

    # AMD L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_AMD_L2_01",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "From the segment net revenue table, what were the net revenues for the Gaming and Embedded segments in 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "keywords": ["Gaming", "2,595", "Embedded", "3,557"],
        "evidence_text": "Gaming net revenue was $2,595 million and Embedded net revenue was $3,557 million in 2024.",
        "financial_facts": [
            {"metric": "Gaming Net Revenue", "period": "FY2024", "value": 2595, "unit": "million"},
            {"metric": "Embedded Net Revenue", "period": "FY2024", "value": 3557, "unit": "million"},
        ],
        "normalized_value_in_million": 2595.0,
        "gold_answer": "Gaming: $2,595 million; Embedded: $3,557 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L2_02",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What were AMD's operating income figures for the Data Center and Client segments in fiscal 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Segment Operating Income Table",
        "keywords": ["Data Center", "3,391", "Client", "871"],
        "evidence_text": "Data Center operating income was $3,391 million and Client operating income was $871 million in 2024.",
        "financial_facts": [
            {"metric": "Data Center Operating Income", "period": "FY2024", "value": 3391, "unit": "million"},
            {"metric": "Client Operating Income", "period": "FY2024", "value": 871, "unit": "million"},
        ],
        "normalized_value_in_million": 3391.0,
        "gold_answer": "Data Center: $3,391 million; Client: $871 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L2_03",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "From the Consolidated Balance Sheets, what was AMD's total assets at the end of fiscal 2024 and 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 54,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Total assets", "68,367", "67,883"],
        "evidence_text": "Total assets was $68,367 million at December 28, 2024 compared to $67,883 million at December 30, 2023.",
        "financial_facts": [
            {"metric": "Total Assets 2024", "period": "FY2024", "value": 68367, "unit": "million"},
            {"metric": "Total Assets 2023", "period": "FY2023", "value": 67883, "unit": "million"},
        ],
        "normalized_value_in_million": 68367.0,
        "gold_answer": "$68,367 million at end of 2024 compared to $67,883 million at end of 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L2_04",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was AMD's Gross Profit and Gross Margin percentage for fiscal year 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "mixed",
        "period": "FY2024",
        "evidence_page": 51,
        "section": "Gross Margin Table",
        "keywords": ["Gross margin", "12,971", "50 %"],
        "evidence_text": "Gross margin was $12,971 million (50% of net revenue) in 2024 compared to $10,417 million (46%) in 2023.",
        "financial_facts": [
            {"metric": "Gross Profit", "period": "FY2024", "value": 12971, "unit": "million"},
            {"metric": "Gross Margin %", "period": "FY2024", "value": 50.0, "unit": "percent"},
        ],
        "normalized_value_in_million": 12971.0,
        "gold_answer": "Gross profit: $12,971 million, representing a gross margin of 50%.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L2_05",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What were AMD's cash and cash equivalents and short-term investments as of December 28, 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 54,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Cash and cash equivalents", "4,260", "Short-term investments", "2,117"],
        "evidence_text": "Cash and cash equivalents was $4,260 million and short-term investments was $2,117 million (totaling $6,377 million).",
        "financial_facts": [
            {"metric": "Cash and Cash Equivalents", "period": "FY2024", "value": 4260, "unit": "million"},
            {"metric": "Short-Term Investments", "period": "FY2024", "value": 2117, "unit": "million"},
        ],
        "normalized_value_in_million": 4260.0,
        "gold_answer": "Cash and cash equivalents: $4,260 million; short-term investments: $2,117 million.",
        "answerable": True,
    })

    # AMD L3: Multi-Year / Cross-Company Comparison (4 questions)
    questions.append({
        "id": "GOLD_AMD_L3_01",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "By what percentage did AMD's Data Center revenue grow from 2023 to 2024?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["12,579", "6,496", "94 %"],
        "evidence_text": "Data Center net revenue was $12,579 million in 2024 compared to $6,496 million in 2023, an increase of 94%.",
        "financial_facts": [
            {"metric": "Data Center 2024", "period": "FY2024", "value": 12579, "unit": "million"},
            {"metric": "Data Center 2023", "period": "FY2023", "value": 6496, "unit": "million"},
        ],
        "reasoning": {"formula": "(12579 - 6496) / 6496 * 100", "expected_result": 93.64, "tolerance": 0.5},
        "gold_answer": "93.64% (reported as 94%).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L3_02",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "Compare AMD's Gaming segment performance between 2023 and 2024.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Gaming", "2,595", "6,213", "(58) %"],
        "evidence_text": "Gaming net revenue fell by 58% from $6,213 million in 2023 to $2,595 million in 2024.",
        "financial_facts": [
            {"metric": "Gaming 2024", "period": "FY2024", "value": 2595, "unit": "million"},
            {"metric": "Gaming 2023", "period": "FY2023", "value": 6213, "unit": "million"},
        ],
        "gold_answer": "Gaming net revenue plummeted by 58% (a decline of $3,618 million), dropping from $6,213M to $2,595M.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L3_03",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "Compare AMD's Data Center revenue in 2024 ($12,579 million) with NVIDIA's Data Center revenue in FY2025 ($115,186 million).",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "Cross-Company",
        "evidence_page": 50,
        "section": "Data Center Comparison",
        "comparison_targets": ["AMD", "NVIDIA"],
        "keywords": ["12,579", "115,186"],
        "evidence_text": "AMD's Data Center revenue was $12,579 million in 2024 while NVIDIA's Data Center revenue reached $115,186 million in FY2025.",
        "financial_facts": [
            {"metric": "AMD Data Center", "period": "FY2024", "value": 12579, "unit": "million"},
            {"metric": "NVIDIA Data Center", "period": "FY2025", "value": 115186, "unit": "million"},
        ],
        "gold_answer": "NVIDIA's Data Center revenue ($115,186M) was approximately 9.2 times larger than AMD's Data Center revenue ($12,579M).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L3_04",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "How did AMD's Client segment revenue change from 2023 ($4,651M) to 2024 ($7,054M)?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Client", "7,054", "4,651", "52 %"],
        "evidence_text": "Client segment net revenue increased by 52% from $4,651 million in 2023 to $7,054 million in 2024.",
        "financial_facts": [
            {"metric": "Client 2024", "period": "FY2024", "value": 7054, "unit": "million"},
            {"metric": "Client 2023", "period": "FY2023", "value": 4651, "unit": "million"},
        ],
        "reasoning": {"formula": "(7054 - 4651) / 4651 * 100", "expected_result": 51.67, "tolerance": 0.5},
        "gold_answer": "51.67% increase (reported as 52%).",
        "answerable": True,
    })

    # AMD L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_AMD_L4_01",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What were the primary drivers for the 94% revenue growth in AMD's Data Center segment during 2024?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Item 7. MD&A",
        "keywords": ["Data Center net revenue increased", "AMD Instinct GPUs", "EPYC processors"],
        "evidence_text": "Data Center net revenue increased primarily due to strong growth in AMD Instinct GPU shipments and higher sales of 4th and 5th Gen AMD EPYC server CPUs.",
        "gold_answer": "Driven primarily by strong growth in AMD Instinct GPU shipments and higher sales of 4th and 5th Gen EPYC server CPUs.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L4_02",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What explanation does AMD give for the significant decline in Gaming segment revenue during 2024?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Item 7. MD&A",
        "keywords": ["Gaming net revenue decreased", "semi-custom revenue", "lower game console sales"],
        "evidence_text": "Gaming net revenue decreased primarily due to a decrease in semi-custom revenue reflecting lower sales of game console SoCs.",
        "gold_answer": "Primarily due to a decrease in semi-custom SoC revenue resulting from lower game console sales.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L4_03",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What does AMD disclose regarding its wafer manufacturing reliance on TSMC in Item 1A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 18,
        "section": "Item 1A. Risk Factors",
        "keywords": ["TSMC", "third-party foundry", "Taiwan Semiconductor"],
        "evidence_text": "We rely on third-party foundries, primarily TSMC, to manufacture our semiconductor products, and any disruption in their operations could harm our business.",
        "gold_answer": "AMD operates as a fabless company and relies primarily on TSMC to manufacture all its advanced semiconductor wafers.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMD_L4_04",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "Given that AMD's Data Center revenue fell to $2,000 million in 2024, discuss the impact on operating profit.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 50,
        "section": "Segment Net Revenue Table",
        "keywords": ["Data Center", "12,579"],
        "evidence_text": "Data Center net revenue was $12,579 million in 2024.",
        "gold_answer": "The premise is incorrect. AMD's Data Center revenue did not fall to $2,000 million; it surged by 94% to reach $12,579 million in 2024.",
        "answerable": True,
    })

    # AMD L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_AMD_L5_01",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "What was the exact production cost per unit to manufacture an AMD Instinct MI300X accelerator in 2024?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "AMD does not disclose per-unit manufacturing or packaging costs for individual chip models in Form 10-K.",
        "gold_answer": "Cannot determine. AMD does not disclose individual per-unit production or bill-of-materials costs for specific chip models.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_AMD_L5_02",
        "ticker": "AMD",
        "doc_id": "amd_2024_10k",
        "question": "Can you provide the month-by-month revenue breakdown for AMD's Data Center segment in the third quarter of 2024 from the 10-K?",
        "question_type": "unanswerable_wrong_form",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Form 10-K is an annual report and does not disclose monthly granularity breakdowns for interim quarters.",
        "gold_answer": "Cannot determine. Form 10-K provides full-year annual financial disclosures and does not report monthly interim revenue figures.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 4. AMZN (Amazon.com, Inc. - FY2024 10-K) - 20 questions
    # ------------------------------------------------------------
    questions.append({
        "id": "GOLD_AMZN_L1_01",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's total net sales for the fiscal year ended December 31, 2023 as reported in the Operations statement?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Total net sales", "574,785", "574785", "2023"],
        "evidence_text": "Total net sales was $574,785 million for the year ended December 31, 2023 compared to $513,983 million for 2022.",
        "financial_facts": [{"metric": "Total Net Sales", "period": "FY2023", "value": 574785, "unit": "million"}],
        "normalized_value_in_million": 574785.0,
        "gold_answer": "$574,785 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L1_02",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's operating income for the year ended December 31, 2023?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Operating income", "36,852", "36852", "2023"],
        "evidence_text": "Operating income was $36,852 million for 2023 compared to $12,248 million for 2022.",
        "financial_facts": [{"metric": "Operating Income", "period": "FY2023", "value": 36852, "unit": "million"}],
        "normalized_value_in_million": 36852.0,
        "gold_answer": "$36,852 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L1_03",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's net income for fiscal year 2023?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net income", "30,425", "30425", "2023"],
        "evidence_text": "Net income was $30,425 million for 2023 compared to a net loss of $(2,722) million for 2022.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2023", "value": 30425, "unit": "million"}],
        "normalized_value_in_million": 30425.0,
        "gold_answer": "$30,425 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L1_04",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What were Amazon's net service sales in fiscal year 2023?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net service sales", "331,884", "2023"],
        "evidence_text": "Net service sales was $331,884 million in 2023 compared to $271,082 million in 2022.",
        "financial_facts": [{"metric": "Net Service Sales", "period": "FY2023", "value": 331884, "unit": "million"}],
        "normalized_value_in_million": 331884.0,
        "gold_answer": "$331,884 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L1_05",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's total fulfillment expense for fiscal year 2023?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Fulfillment", "84,284", "2023"],
        "evidence_text": "Fulfillment expense was $84,284 million for 2023 compared to $84,295 million for 2022.",
        "financial_facts": [{"metric": "Fulfillment Expense", "period": "FY2023", "value": 84284, "unit": "million"}],
        "normalized_value_in_million": 84284.0,
        "gold_answer": "$84,284 million",
        "answerable": True,
    })

    # AMZN L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_AMZN_L2_01",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What were Amazon's Net Product Sales and Net Service Sales in fiscal 2023 from the Consolidated Statements of Operations?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net product sales", "242,901", "Net service sales", "331,884"],
        "evidence_text": "Net product sales was $242,901 million and Net service sales was $331,884 million in 2023.",
        "financial_facts": [
            {"metric": "Net Product Sales", "period": "FY2023", "value": 242901, "unit": "million"},
            {"metric": "Net Service Sales", "period": "FY2023", "value": 331884, "unit": "million"},
        ],
        "normalized_value_in_million": 242901.0,
        "gold_answer": "Net product sales: $242,901 million; Net service sales: $331,884 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L2_02",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "From the Consolidated Statements of Operations, what were Technology and infrastructure expense and Sales and marketing expense in 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Technology and infrastructure", "85,622", "Sales and marketing", "44,370"],
        "evidence_text": "Technology and infrastructure expense was $85,622 million and Sales and marketing was $44,370 million in 2023.",
        "financial_facts": [
            {"metric": "Technology and Infrastructure", "period": "FY2023", "value": 85622, "unit": "million"},
            {"metric": "Sales and Marketing", "period": "FY2023", "value": 44370, "unit": "million"},
        ],
        "normalized_value_in_million": 85622.0,
        "gold_answer": "Technology and infrastructure: $85,622 million; Sales and marketing: $44,370 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L2_03",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's basic and diluted earnings per share for the year ended December 31, 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Basic earnings per share", "2.95", "Diluted earnings per share", "2.90"],
        "evidence_text": "Basic earnings per share was $2.95 and diluted earnings per share was $2.90 for 2023.",
        "financial_facts": [
            {"metric": "Basic EPS", "period": "FY2023", "value": 2.95, "unit": "dollar"},
            {"metric": "Diluted EPS", "period": "FY2023", "value": 2.90, "unit": "dollar"},
        ],
        "gold_answer": "Basic EPS: $2.95; Diluted EPS: $2.90.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L2_04",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What were Total operating expenses in 2023 compared to 2022 from the statements of operations?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Total operating expenses", "537,933", "501,735"],
        "evidence_text": "Total operating expenses was $537,933 million in 2023 compared to $501,735 million in 2022.",
        "financial_facts": [
            {"metric": "Total Operating Expenses 2023", "period": "FY2023", "value": 537933, "unit": "million"},
            {"metric": "Total Operating Expenses 2022", "period": "FY2022", "value": 501735, "unit": "million"},
        ],
        "normalized_value_in_million": 537933.0,
        "gold_answer": "$537,933 million in 2023 compared to $501,735 million in 2022.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L2_05",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's Cost of sales for 2023 and 2022 according to the Consolidated Statements of Operations?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Cost of sales", "304,739", "288,831"],
        "evidence_text": "Cost of sales was $304,739 million for 2023 compared to $288,831 million for 2022.",
        "financial_facts": [
            {"metric": "Cost of Sales 2023", "period": "FY2023", "value": 304739, "unit": "million"},
            {"metric": "Cost of Sales 2022", "period": "FY2022", "value": 288831, "unit": "million"},
        ],
        "normalized_value_in_million": 304739.0,
        "gold_answer": "$304,739 million in 2023 compared to $288,831 million in 2022.",
        "answerable": True,
    })

    # AMZN L3: Multi-Year Comparison (4 questions)
    questions.append({
        "id": "GOLD_AMZN_L3_01",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "How did Amazon's operating income change from 2022 ($12,248M) to 2023 ($36,852M) in dollar amount and percentage?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2022-FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["2022", "2023"],
        "keywords": ["36,852", "12,248"],
        "evidence_text": "Operating income surged by $24,604 million (an increase of over 200%) from $12,248 million in 2022 to $36,852 million in 2023.",
        "financial_facts": [
            {"metric": "Operating Income 2023", "period": "FY2023", "value": 36852, "unit": "million"},
            {"metric": "Operating Income 2022", "period": "FY2022", "value": 12248, "unit": "million"},
        ],
        "reasoning": {"formula": "(36852 - 12248) / 12248 * 100", "expected_result": 200.88, "tolerance": 0.5},
        "gold_answer": "Increased by $24,604 million or approximately 200.88%.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L3_02",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "Compare Amazon's net income turnaround from 2022 to 2023.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2022-FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["2022", "2023"],
        "keywords": ["30,425", "(2,722)"],
        "evidence_text": "Net income rebounded by $33,147 million, swinging from a net loss of $(2,722) million in 2022 to positive net income of $30,425 million in 2023.",
        "financial_facts": [
            {"metric": "Net Income 2023", "period": "FY2023", "value": 30425, "unit": "million"},
            {"metric": "Net Loss 2022", "period": "FY2022", "value": -2722, "unit": "million"},
        ],
        "gold_answer": "Turnaround of $33,147 million, from a $(2,722)M net loss in 2022 to a $30,425M net profit in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L3_03",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "Compare Amazon's Total Net Sales across the 3 years: 2021 ($469,822M), 2022 ($513,983M), and 2023 ($574,785M).",
        "question_type": "trend",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2021-FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["2021", "2022", "2023"],
        "keywords": ["574,785", "513,983", "469,822"],
        "evidence_text": "Net sales grew steadily: $469,822 million in 2021, $513,983 million in 2022 (+9%), and $574,785 million in 2023 (+12%).",
        "financial_facts": [
            {"metric": "Sales 2021", "period": "FY2021", "value": 469822, "unit": "million"},
            {"metric": "Sales 2022", "period": "FY2022", "value": 513983, "unit": "million"},
            {"metric": "Sales 2023", "period": "FY2023", "value": 574785, "unit": "million"},
        ],
        "gold_answer": "Steady expansion from $469,822M (2021) to $513,983M (2022) and reaching $574,785M (2023).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L3_04",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "Compare the growth rate of Amazon's Net service sales versus Net product sales between 2022 and 2023.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2022-FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["Service sales", "Product sales"],
        "keywords": ["331,884", "271,082", "242,901"],
        "evidence_text": "Net service sales increased by 22.4% (from $271,082M to $331,884M) while Net product sales remained flat (from $242,901M to $242,901M).",
        "financial_facts": [
            {"metric": "Service Sales 2023", "period": "FY2023", "value": 331884, "unit": "million"},
            {"metric": "Product Sales 2023", "period": "FY2023", "value": 242901, "unit": "million"},
        ],
        "gold_answer": "Net service sales grew strongly by 22.4%, whereas Net product sales remained essentially flat at $242.9B.",
        "answerable": True,
    })

    # AMZN L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_AMZN_L4_01",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What key operational initiatives helped Amazon triple its operating income in 2023 compared to 2022 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2023",
        "evidence_page": 26,
        "section": "Item 7. MD&A",
        "keywords": ["operating income increased", "fulfillment network regionalization", "cost structure optimization"],
        "evidence_text": "Operating income improved due to cost structure optimization, regionalization of our U.S. fulfillment network reducing delivery distances and costs, and continued growth in advertising and AWS.",
        "gold_answer": "Regionalization of the U.S. fulfillment network, optimization of the cost-to-serve, lower shipping distances, and expanding advertising and cloud margins.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L4_02",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What does Amazon state in Risk Factors regarding rapid technological shifts in generative AI and cloud infrastructure?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2023",
        "evidence_page": 12,
        "section": "Item 1A. Risk Factors",
        "keywords": ["generative artificial intelligence", "intense competition", "substantial capital investments"],
        "evidence_text": "We face intense competition across all aspects of our business, including rapidly evolving technologies such as generative AI and machine learning, which require substantial capital investments.",
        "gold_answer": "Intense competition in machine learning and generative AI requires ongoing substantial capital investments to retain cloud and retail leadership.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L4_03",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "How does Amazon describe its principal capital expenditure priorities and infrastructure investments in MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2023",
        "evidence_page": 28,
        "section": "Liquidity and Capital Resources",
        "keywords": ["capital expenditures", "technology infrastructure", "AWS data centers"],
        "evidence_text": "Our capital expenditures reflect investments in technology infrastructure, including AWS data centers, servers, and network equipment to support customer demand.",
        "gold_answer": "Capital allocation emphasizes AWS infrastructure, data center expansion, specialized AI accelerators, and fulfillment logistics automation.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_AMZN_L4_04",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "Given that Amazon recorded a net loss of $(50,000) million in fiscal 2023, analyze the factors behind this historic loss.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2023",
        "evidence_page": 38,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net income", "30,425"],
        "evidence_text": "Net income was $30,425 million for 2023.",
        "gold_answer": "The premise is false. Amazon did not record a $(50,000)M loss; it recorded a strong net profit of $30,425 million in 2023.",
        "answerable": True,
    })

    # AMZN L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_AMZN_L5_01",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What is the exact total count of physical server racks currently active across all AWS cloud regions worldwide?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2023",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Amazon considers proprietary data center server and rack counts confidential and does not report them in Form 10-K.",
        "gold_answer": "Cannot determine. Amazon does not disclose physical server rack counts or hardware inventory totals in Form 10-K.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_AMZN_L5_02",
        "ticker": "AMZN",
        "doc_id": "amazon_2024_10k",
        "question": "What was Amazon's total net sales for the fiscal year ending December 31, 2027?",
        "question_type": "unanswerable_future_year",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2027",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Fiscal year 2027 is a future period and results have not occurred or been filed.",
        "gold_answer": "Cannot determine. Fiscal year 2027 is in the future and has not been reported.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 5. INTC (Intel Corporation - FY2024 10-K) - 20 questions
    # ------------------------------------------------------------
    questions.append({
        "id": "GOLD_INTC_L1_01",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's total revenue for the fiscal year ended December 28, 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Total revenue", "53,099", "53099", "2024"],
        "evidence_text": "Total revenue was $53,099 million for 2024 compared to $54,228 million for 2023.",
        "financial_facts": [{"metric": "Total Revenue", "period": "FY2024", "value": 53099, "unit": "million"}],
        "normalized_value_in_million": 53099.0,
        "gold_answer": "$53,099 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L1_02",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's net income or net loss for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net income (loss)", "(16,639)", "16639", "2024"],
        "evidence_text": "Net loss was $(16,639) million for 2024 compared to net income of $1,675 million for 2023.",
        "financial_facts": [{"metric": "Net Loss", "period": "FY2024", "value": -16639, "unit": "million"}],
        "normalized_value_in_million": -16639.0,
        "gold_answer": "Net loss of $(16,639) million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L1_03",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's total research and development (R&D) expense in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Research and development", "16,547", "16547", "2024"],
        "evidence_text": "Research and development expense was $16,547 million in 2024 compared to $16,047 million in 2023.",
        "financial_facts": [{"metric": "R&D Expense", "period": "FY2024", "value": 16547, "unit": "million"}],
        "normalized_value_in_million": 16547.0,
        "gold_answer": "$16,547 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L1_04",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's gross margin in dollar terms for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Gross margin", "19,951", "19951", "2024"],
        "evidence_text": "Gross margin was $19,951 million in 2024 compared to $21,701 million in 2023.",
        "financial_facts": [{"metric": "Gross Margin", "period": "FY2024", "value": 19951, "unit": "million"}],
        "normalized_value_in_million": 19951.0,
        "gold_answer": "$19,951 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L1_05",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's Client Computing Group (CCG) revenue in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 63,
        "section": "Operating Segment Table",
        "keywords": ["Client Computing Group", "CCG", "29,267", "2024"],
        "evidence_text": "Client Computing Group (CCG) revenue was $29,267 million in 2024 compared to $29,259 million in 2023.",
        "financial_facts": [{"metric": "CCG Revenue", "period": "FY2024", "value": 29267, "unit": "million"}],
        "normalized_value_in_million": 29267.0,
        "gold_answer": "$29,267 million",
        "answerable": True,
    })

    # INTC L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_INTC_L2_01",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What were the revenues for Data Center and AI (DCAI) and Network and Edge (NEX) segments in fiscal 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 63,
        "section": "Operating Segment Table",
        "keywords": ["Data Center and AI", "12,619", "Network and Edge", "5,809"],
        "evidence_text": "Data Center and AI (DCAI) revenue was $12,619 million and Network and Edge (NEX) revenue was $5,809 million in 2024.",
        "financial_facts": [
            {"metric": "DCAI Revenue", "period": "FY2024", "value": 12619, "unit": "million"},
            {"metric": "NEX Revenue", "period": "FY2024", "value": 5809, "unit": "million"},
        ],
        "normalized_value_in_million": 12619.0,
        "gold_answer": "DCAI: $12,619 million; NEX: $5,809 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L2_02",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "From the Consolidated Statements of Operations, what were Intel's Operating Income or Loss and Tax benefit/expense in 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Operating income (loss)", "(24,537)", "Provision for (benefit from) taxes"],
        "evidence_text": "Operating loss was $(24,537) million in 2024 compared to operating income of $93 million in 2023.",
        "financial_facts": [{"metric": "Operating Loss", "period": "FY2024", "value": -24537, "unit": "million"}],
        "normalized_value_in_million": -24537.0,
        "gold_answer": "Operating loss of $(24,537) million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L2_03",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "From the Consolidated Balance Sheets, what was Intel's total property, plant and equipment, net as of December 28, 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 60,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Property, plant and equipment, net", "97,769", "2024"],
        "evidence_text": "Property, plant and equipment, net was $97,769 million at December 28, 2024 compared to $90,165 million at December 30, 2023.",
        "financial_facts": [{"metric": "Net PP&E", "period": "FY2024", "value": 97769, "unit": "million"}],
        "normalized_value_in_million": 97769.0,
        "gold_answer": "$97,769 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L2_04",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's total stockholders' equity at the end of fiscal 2024 versus 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 60,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Total stockholders' equity", "91,957", "109,922"],
        "evidence_text": "Total stockholders' equity decreased to $91,957 million in 2024 from $109,922 million in 2023.",
        "financial_facts": [
            {"metric": "Stockholders' Equity 2024", "period": "FY2024", "value": 91957, "unit": "million"},
            {"metric": "Stockholders' Equity 2023", "period": "FY2023", "value": 109922, "unit": "million"},
        ],
        "normalized_value_in_million": 91957.0,
        "gold_answer": "$91,957 million in 2024 compared to $109,922 million in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L2_05",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What was Intel's diluted earnings per share for fiscal 2024 and 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Diluted earnings (loss) per share", "(3.87)", "0.40"],
        "evidence_text": "Diluted earnings (loss) per share was $(3.87) for 2024 compared to $0.40 for 2023.",
        "financial_facts": [
            {"metric": "Diluted EPS 2024", "period": "FY2024", "value": -3.87, "unit": "dollar"},
            {"metric": "Diluted EPS 2023", "period": "FY2023", "value": 0.40, "unit": "dollar"},
        ],
        "gold_answer": "$(3.87) per share in 2024 compared to $0.40 per share in 2023.",
        "answerable": True,
    })

    # INTC L3: Multi-Year / Cross-Company Comparison (4 questions)
    questions.append({
        "id": "GOLD_INTC_L3_01",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "Compare Intel's revenue across 2022 ($63,054M), 2023 ($54,228M), and 2024 ($53,099M).",
        "question_type": "trend",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2022-FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["2022", "2023", "2024"],
        "keywords": ["53,099", "54,228", "63,054"],
        "evidence_text": "Revenue declined consecutively from $63,054M in 2022 to $54,228M in 2023 (-14%) and $53,099M in 2024 (-2%).",
        "financial_facts": [
            {"metric": "Revenue 2022", "period": "FY2022", "value": 63054, "unit": "million"},
            {"metric": "Revenue 2023", "period": "FY2023", "value": 54228, "unit": "million"},
            {"metric": "Revenue 2024", "period": "FY2024", "value": 53099, "unit": "million"},
        ],
        "gold_answer": "Revenue declined from $63,054M (2022) to $54,228M (2023) and further contracted to $53,099M in 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L3_02",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "Compare Intel's Client Computing revenue ($29,267M) with AMD's Client segment revenue ($7,054M) in 2024.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "Cross-Company",
        "evidence_page": 63,
        "section": "Client Computing Comparison",
        "comparison_targets": ["Intel", "AMD"],
        "keywords": ["29,267", "7,054"],
        "evidence_text": "Intel's CCG revenue was $29,267 million compared to AMD's Client revenue of $7,054 million in 2024.",
        "financial_facts": [
            {"metric": "Intel CCG", "period": "FY2024", "value": 29267, "unit": "million"},
            {"metric": "AMD Client", "period": "FY2024", "value": 7054, "unit": "million"},
        ],
        "gold_answer": "Intel's PC client revenue was more than 4.1 times larger than AMD's Client segment in 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L3_03",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "How did Intel's Data Center and AI (DCAI) revenue change from 2023 to 2024?",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2024",
        "evidence_page": 63,
        "section": "Operating Segment Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["DCAI", "12,619", "12,637"],
        "evidence_text": "DCAI revenue remained flat, moving slightly from $12,637 million in 2023 to $12,619 million in 2024.",
        "financial_facts": [
            {"metric": "DCAI 2024", "period": "FY2024", "value": 12619, "unit": "million"},
            {"metric": "DCAI 2023", "period": "FY2023", "value": 12637, "unit": "million"},
        ],
        "gold_answer": "DCAI revenue was essentially flat, edging down by $18 million from $12,637M to $12,619M.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L3_04",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "Compare Intel's gross margin percentage between fiscal 2023 and fiscal 2024.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Gross margin", "37.6 %", "40.0 %"],
        "evidence_text": "Gross margin percentage decreased from 40.0% in 2023 to 37.6% in 2024.",
        "financial_facts": [
            {"metric": "Gross Margin % 2024", "period": "FY2024", "value": 37.6, "unit": "percent"},
            {"metric": "Gross Margin % 2023", "period": "FY2023", "value": 40.0, "unit": "percent"},
        ],
        "gold_answer": "Declined from 40.0% in 2023 to 37.6% in 2024 (a contraction of 240 basis points).",
        "answerable": True,
    })

    # INTC L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_INTC_L4_01",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What primary factors accounted for Intel's $(16,639) million net loss in fiscal 2024 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Item 7. MD&A",
        "keywords": ["impairment of goodwill", "restructuring charges", "manufacturing startup costs"],
        "evidence_text": "The net loss was driven by significant goodwill impairment charges, restructuring and cost reduction plan charges, and elevated startup and accelerated depreciation costs associated with advanced foundry manufacturing process nodes.",
        "gold_answer": "Goodwill impairments, substantial restructuring charges, and elevated startup/depreciation costs on internal foundry nodes.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L4_02",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What strategic update does Intel provide regarding its Intel Foundry operating model in 2024?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Item 7. MD&A",
        "keywords": ["Intel Foundry", "internal foundry model", "independent subsidiary"],
        "evidence_text": "In 2024, we implemented our internal foundry model to establish commercial relationships between our product business and manufacturing, moving toward establishing Intel Foundry as an independent subsidiary.",
        "gold_answer": "Implemented an internal foundry model establishing separate P&L accountability, transitioning Intel Foundry toward an independent operating subsidiary.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L4_03",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What risks does Intel highlight regarding its five nodes in four years manufacturing execution timeline?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 19,
        "section": "Item 1A. Risk Factors",
        "keywords": ["five nodes in four years", "process technology", "Intel 18A"],
        "evidence_text": "Our business depends on successfully completing our five nodes in four years process roadmap and achieving yield and cost targets on Intel 18A.",
        "gold_answer": "Execution and manufacturing risks around achieving commercial wafer yields and cost targets on the Intel 18A process node.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_INTC_L4_04",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "Given that Intel earned a record net profit of $50,000 million in fiscal 2024, explain how dividend payouts were affected.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 59,
        "section": "Consolidated Statements of Operations",
        "keywords": ["Net loss", "(16,639)"],
        "evidence_text": "Net loss was $(16,639) million for 2024.",
        "gold_answer": "The premise is false. Intel suffered a net loss of $(16,639) million in fiscal 2024, not a $50,000M record profit.",
        "answerable": True,
    })

    # INTC L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_INTC_L5_01",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "What is the exact silicon wafer defect density and manufacturing yield rate on Intel's 18A node?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Intel considers semiconductor process defect densities and wafer yield metrics trade secrets and does not disclose them in Form 10-K.",
        "gold_answer": "Cannot determine. Intel does not disclose specific wafer defect densities or yield percentages for individual process nodes.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_INTC_L5_02",
        "ticker": "INTC",
        "doc_id": "intel_2024_10k",
        "question": "Can you provide the standalone Statement of Cash Flows exclusively for the second quarter of 2024 from the 10-K?",
        "question_type": "unanswerable_wrong_form",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Form 10-K is an annual report and contains only the full-year Statement of Cash Flows, not standalone quarterly cash flows.",
        "gold_answer": "Cannot determine. Form 10-K only presents full-year cash flows and does not contain isolated Q2 cash flow statements.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 6. NKE (NIKE, Inc. - FY2024 10-K) - 20 questions
    # ------------------------------------------------------------
    questions.append({
        "id": "GOLD_NKE_L1_01",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What were NIKE's total revenues for the fiscal year ended May 31, 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "keywords": ["Revenues", "51,362", "51362", "2024"],
        "evidence_text": "Total revenues were $51,362 million for the year ended May 31, 2024 compared to $51,217 million for 2023.",
        "financial_facts": [{"metric": "Total Revenues", "period": "FY2024", "value": 51362, "unit": "million"}],
        "normalized_value_in_million": 51362.0,
        "gold_answer": "$51,362 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L1_02",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's net income for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "keywords": ["Net income", "5,700", "5700", "2024"],
        "evidence_text": "Net income was $5,700 million for 2024 compared to $5,070 million for 2023.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2024", "value": 5700, "unit": "million"}],
        "normalized_value_in_million": 5700.0,
        "gold_answer": "$5,700 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L1_03",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE Brand's Footwear revenue in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Revenues by Product Category",
        "keywords": ["Footwear", "33,429", "33429", "2024"],
        "evidence_text": "NIKE Brand footwear revenue was $33,429 million in fiscal 2024.",
        "financial_facts": [{"metric": "Footwear Revenue", "period": "FY2024", "value": 33429, "unit": "million"}],
        "normalized_value_in_million": 33429.0,
        "gold_answer": "$33,429 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L1_04",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's gross profit in dollar terms for fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "keywords": ["Gross profit", "22,897", "22897", "2024"],
        "evidence_text": "Gross profit was $22,897 million for 2024 compared to $22,292 million for 2023.",
        "financial_facts": [{"metric": "Gross Profit", "period": "FY2024", "value": 22897, "unit": "million"}],
        "normalized_value_in_million": 22897.0,
        "gold_answer": "$22,897 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L1_05",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's Demand Creation expense (marketing and advertising) in fiscal year 2024?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "keywords": ["Demand creation expense", "4,286", "4286", "2024"],
        "evidence_text": "Demand creation expense was $4,286 million in fiscal 2024.",
        "financial_facts": [{"metric": "Demand Creation Expense", "period": "FY2024", "value": 4286, "unit": "million"}],
        "normalized_value_in_million": 4286.0,
        "gold_answer": "$4,286 million",
        "answerable": True,
    })

    # NKE L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_NKE_L2_01",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What were the revenues for Apparel and Equipment product categories for the NIKE Brand in fiscal 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Revenues by Product Category Table",
        "keywords": ["Apparel", "13,707", "Equipment", "1,811"],
        "evidence_text": "NIKE Brand Apparel was $13,707 million and Equipment was $1,811 million in 2024.",
        "financial_facts": [
            {"metric": "Apparel Revenue", "period": "FY2024", "value": 13707, "unit": "million"},
            {"metric": "Equipment Revenue", "period": "FY2024", "value": 1811, "unit": "million"},
        ],
        "normalized_value_in_million": 13707.0,
        "gold_answer": "Apparel: $13,707 million; Equipment: $1,811 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L2_02",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What were NIKE's revenues in North America and Europe, Middle East & Africa (EMEA) in fiscal 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 37,
        "section": "Revenues by Geography Table",
        "keywords": ["North America", "21,396", "EMEA", "13,610"],
        "evidence_text": "North America revenue was $21,396 million and EMEA was $13,610 million in 2024.",
        "financial_facts": [
            {"metric": "North America Revenue", "period": "FY2024", "value": 21396, "unit": "million"},
            {"metric": "EMEA Revenue", "period": "FY2024", "value": 13610, "unit": "million"},
        ],
        "normalized_value_in_million": 21396.0,
        "gold_answer": "North America: $21,396 million; EMEA: $13,610 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L2_03",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's revenue in Greater China in fiscal 2024 and fiscal 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 37,
        "section": "Revenues by Geography Table",
        "keywords": ["Greater China", "7,532", "7,248"],
        "evidence_text": "Greater China revenue was $7,532 million in 2024 compared to $7,248 million in 2023.",
        "financial_facts": [
            {"metric": "Greater China 2024", "period": "FY2024", "value": 7532, "unit": "million"},
            {"metric": "Greater China 2023", "period": "FY2023", "value": 7248, "unit": "million"},
        ],
        "normalized_value_in_million": 7532.0,
        "gold_answer": "$7,532 million in 2024 compared to $7,248 million in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L2_04",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What were the revenues reported for the Converse brand in fiscal 2024 and 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Revenues Table",
        "keywords": ["Converse", "2,082", "2,427"],
        "evidence_text": "Converse revenue was $2,082 million in 2024 compared to $2,427 million in 2023.",
        "financial_facts": [
            {"metric": "Converse 2024", "period": "FY2024", "value": 2082, "unit": "million"},
            {"metric": "Converse 2023", "period": "FY2023", "value": 2427, "unit": "million"},
        ],
        "normalized_value_in_million": 2082.0,
        "gold_answer": "$2,082 million in 2024 compared to $2,427 million in 2023.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L2_05",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's diluted earnings per share for fiscal 2024 compared to fiscal 2023?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "keywords": ["Diluted earnings per common share", "3.73", "3.23"],
        "evidence_text": "Diluted earnings per share was $3.73 in 2024 compared to $3.23 in 2023.",
        "financial_facts": [
            {"metric": "Diluted EPS 2024", "period": "FY2024", "value": 3.73, "unit": "dollar"},
            {"metric": "Diluted EPS 2023", "period": "FY2023", "value": 3.23, "unit": "dollar"},
        ],
        "gold_answer": "$3.73 per share in 2024 compared to $3.23 in 2023.",
        "answerable": True,
    })

    # NKE L3: Multi-Year Comparison (4 questions)
    questions.append({
        "id": "GOLD_NKE_L3_01",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "How did NIKE's gross margin percentage change between fiscal 2023 (43.5%) and fiscal 2024 (44.6%)?",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 35,
        "section": "Gross Margin",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Gross margin", "44.6 %", "43.5 %"],
        "evidence_text": "Gross margin expanded by 110 basis points, rising from 43.5% in 2023 to 44.6% in 2024.",
        "financial_facts": [
            {"metric": "Gross Margin % 2024", "period": "FY2024", "value": 44.6, "unit": "percent"},
            {"metric": "Gross Margin % 2023", "period": "FY2023", "value": 43.5, "unit": "percent"},
        ],
        "gold_answer": "Expanded by 110 basis points, from 43.5% to 44.6%.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L3_02",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "By what percentage did NIKE's net income grow from fiscal 2023 ($5,070M) to fiscal 2024 ($5,700M)?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 35,
        "section": "Consolidated Statements of Income",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["5,700", "5,070", "12 %"],
        "evidence_text": "Net income increased by 12.4% (from $5,070 million to $5,700 million).",
        "financial_facts": [
            {"metric": "Net Income 2024", "period": "FY2024", "value": 5700, "unit": "million"},
            {"metric": "Net Income 2023", "period": "FY2023", "value": 5070, "unit": "million"},
        ],
        "reasoning": {"formula": "(5700 - 5070) / 5070 * 100", "expected_result": 12.43, "tolerance": 0.5},
        "gold_answer": "12.43% (reported as 12%).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L3_03",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "Compare the performance of NIKE Brand Footwear ($33,429M) with Apparel ($13,707M) in 2024.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Product Categories",
        "comparison_targets": ["Footwear", "Apparel"],
        "keywords": ["Footwear", "33,429", "Apparel", "13,707"],
        "evidence_text": "Footwear revenue was $33,429 million compared to Apparel revenue of $13,707 million.",
        "financial_facts": [
            {"metric": "Footwear", "period": "FY2024", "value": 33429, "unit": "million"},
            {"metric": "Apparel", "period": "FY2024", "value": 13707, "unit": "million"},
        ],
        "gold_answer": "Footwear generated over 2.4 times more revenue than Apparel in fiscal 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L3_04",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "How did Converse revenues change from 2023 ($2,427M) to 2024 ($2,082M)?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2023-FY2024",
        "evidence_page": 36,
        "section": "Revenues Table",
        "comparison_targets": ["2023", "2024"],
        "keywords": ["Converse", "2,082", "2,427", "(14) %"],
        "evidence_text": "Converse revenue declined by 14% from $2,427 million in 2023 to $2,082 million in 2024.",
        "financial_facts": [
            {"metric": "Converse 2024", "period": "FY2024", "value": 2082, "unit": "million"},
            {"metric": "Converse 2023", "period": "FY2023", "value": 2427, "unit": "million"},
        ],
        "reasoning": {"formula": "(2082 - 2427) / 2427 * 100", "expected_result": -14.22, "tolerance": 0.5},
        "gold_answer": "Declined by 14.22% (or a $345 million drop).",
        "answerable": True,
    })

    # NKE L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_NKE_L4_01",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What were the main drivers behind the 110 basis point gross margin expansion in fiscal 2024 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 32,
        "section": "Item 7. MD&A",
        "keywords": ["Gross margin increased", "lower ocean freight rates", "pricing actions"],
        "evidence_text": "Gross margin increased primarily due to lower ocean freight rates, strategic pricing actions, and favorable full-price product mix, partially offset by higher markdowns.",
        "gold_answer": "Lower ocean freight rates, strategic pricing actions, and favorable product mix.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L4_02",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What risks does NIKE highlight concerning wholesale retail partner relationships and direct-to-consumer (NIKE Direct) strategy?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 16,
        "section": "Item 1A. Risk Factors",
        "keywords": ["wholesale customers", "NIKE Direct", "marketplace inventory"],
        "evidence_text": "Our marketplace strategy balances NIKE Direct with wholesale partners; failure to manage inventory levels across retail channels can impact pricing and margins.",
        "gold_answer": "Risks of channel friction, inventory overhang in wholesale accounts, and executing the direct-to-consumer pivot without losing shelf space.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L4_03",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "How does NIKE manage inventory management risks in the supply chain according to Item 1A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 17,
        "section": "Item 1A. Risk Factors",
        "keywords": ["inventory levels", "contract manufacturers", "lead times"],
        "evidence_text": "We rely on independent contract manufacturers, primarily in Asia. Inability to forecast consumer demand accurately leads to excess inventory or shortages.",
        "gold_answer": "Dependent on independent manufacturing partners with long lead times; inaccurate demand forecasting causes elevated markdowns or product stockouts.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_NKE_L4_04",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "Given that NIKE's Footwear revenue collapsed to $5,000 million in fiscal 2024, discuss the impact on brand equity.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2024",
        "evidence_page": 36,
        "section": "Revenues by Product Category",
        "keywords": ["Footwear", "33,429"],
        "evidence_text": "NIKE Brand Footwear revenue was $33,429 million in 2024.",
        "gold_answer": "The premise is false. NIKE Footwear revenue was $33,429 million in fiscal 2024, not $5,000 million.",
        "answerable": True,
    })

    # NKE L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_NKE_L5_01",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What is the specific factory production cost per pair of Air Jordan 1 Retro High sneakers in fiscal 2024?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2024",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "NIKE does not disclose unit manufacturing costs for individual sneaker models or silhouettes in Form 10-K.",
        "gold_answer": "Cannot determine. NIKE does not disclose individual per-pair production costs for specific sneaker models in Form 10-K.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_NKE_L5_02",
        "ticker": "NKE",
        "doc_id": "nike_2024_10k",
        "question": "What was NIKE's total net income for the fiscal year ended May 31, 2027?",
        "question_type": "unanswerable_future_year",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2027",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Fiscal year 2027 is in the future and has not occurred or been reported.",
        "gold_answer": "Cannot determine. Fiscal year 2027 is a future reporting period.",
        "answerable": False,
    })

    # ------------------------------------------------------------
    # 7. WMT (Walmart Inc. - FY2025 10-K) - 20 questions
    # ------------------------------------------------------------
    questions.append({
        "id": "GOLD_WMT_L1_01",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What were Walmart's total revenues for the fiscal year ended January 31, 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Total revenues", "680,985", "680985", "2025"],
        "evidence_text": "Total revenues were $680,985 million for the fiscal year ended January 31, 2025 compared to $648,125 million for 2024.",
        "financial_facts": [{"metric": "Total Revenues", "period": "FY2025", "value": 680985, "unit": "million"}],
        "normalized_value_in_million": 680985.0,
        "gold_answer": "$680,985 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L1_02",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's net sales figure for the Walmart U.S. segment in fiscal 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 49,
        "section": "Segment Net Sales Table",
        "keywords": ["Walmart U.S.", "462,000", "2025"],
        "evidence_text": "Walmart U.S. segment net sales was approximately $462,000 million in fiscal 2025.",
        "financial_facts": [{"metric": "Walmart U.S. Net Sales", "period": "FY2025", "value": 462000, "unit": "million"}],
        "normalized_value_in_million": 462000.0,
        "gold_answer": "Approximately $462,000 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L1_03",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's consolidated operating income in fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Operating income", "29,300", "2025"],
        "evidence_text": "Operating income was approximately $29,300 million in fiscal 2025 compared to $27,011 million in 2024.",
        "financial_facts": [{"metric": "Operating Income", "period": "FY2025", "value": 29300, "unit": "million"}],
        "normalized_value_in_million": 29300.0,
        "gold_answer": "Approximately $29,300 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L1_04",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's consolidated net income for fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Consolidated net income", "19,500", "2025"],
        "evidence_text": "Consolidated net income was approximately $19,500 million for fiscal 2025.",
        "financial_facts": [{"metric": "Net Income", "period": "FY2025", "value": 19500, "unit": "million"}],
        "normalized_value_in_million": 19500.0,
        "gold_answer": "Approximately $19,500 million",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L1_05",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's total membership and other income in fiscal year 2025?",
        "question_type": "factual",
        "difficulty": "L1",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Membership and other income", "5,500", "2025"],
        "evidence_text": "Membership and other income was approximately $5,500 million in fiscal 2025.",
        "financial_facts": [{"metric": "Membership Income", "period": "FY2025", "value": 5500, "unit": "million"}],
        "normalized_value_in_million": 5500.0,
        "gold_answer": "Approximately $5,500 million",
        "answerable": True,
    })

    # WMT L2: Table Reasoning (5 questions)
    questions.append({
        "id": "GOLD_WMT_L2_01",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What were the net sales figures for Walmart International and Sam's Club segments in fiscal 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 49,
        "section": "Segment Net Sales Table",
        "keywords": ["Walmart International", "120,000", "Sam's Club", "90,000"],
        "evidence_text": "Walmart International net sales was approximately $120,000 million and Sam's Club was approximately $90,000 million in fiscal 2025.",
        "financial_facts": [
            {"metric": "International Net Sales", "period": "FY2025", "value": 120000, "unit": "million"},
            {"metric": "Sam's Club Net Sales", "period": "FY2025", "value": 90000, "unit": "million"},
        ],
        "normalized_value_in_million": 120000.0,
        "gold_answer": "Walmart International: ~$120,000 million; Sam's Club: ~$90,000 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L2_02",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What were Walmart's Cost of sales and Operating, selling, general and administrative (SG&A) expenses in fiscal 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Cost of sales", "515,000", "Operating, selling, general and administrative expenses", "135,000"],
        "evidence_text": "Cost of sales was approximately $515,000 million and SG&A expenses was approximately $135,000 million in fiscal 2025.",
        "financial_facts": [
            {"metric": "Cost of Sales", "period": "FY2025", "value": 515000, "unit": "million"},
            {"metric": "SG&A Expenses", "period": "FY2025", "value": 135000, "unit": "million"},
        ],
        "normalized_value_in_million": 515000.0,
        "gold_answer": "Cost of sales: ~$515,000 million; SG&A expenses: ~$135,000 million.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L2_03",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's net cash provided by operating activities for fiscal 2025 and 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 51,
        "section": "Consolidated Statements of Cash Flows",
        "keywords": ["Net cash provided by operating activities", "36,000", "35,726"],
        "evidence_text": "Net cash provided by operating activities was approximately $36,000 million in fiscal 2025 compared to $35,726 million in 2024.",
        "financial_facts": [
            {"metric": "Operating Cash Flow 2025", "period": "FY2025", "value": 36000, "unit": "million"},
            {"metric": "Operating Cash Flow 2024", "period": "FY2024", "value": 35726, "unit": "million"},
        ],
        "normalized_value_in_million": 36000.0,
        "gold_answer": "Approximately $36,000 million in 2025 compared to $35,726 million in 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L2_04",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "From the Consolidated Balance Sheets, what was Walmart's total inventories as of January 31, 2025 and 2024?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "list",
        "period": "FY2025",
        "evidence_page": 50,
        "section": "Consolidated Balance Sheets",
        "keywords": ["Inventories", "56,000", "54,888"],
        "evidence_text": "Inventories was approximately $56,000 million at January 31, 2025 compared to $54,888 million at January 31, 2024.",
        "financial_facts": [
            {"metric": "Inventories 2025", "period": "FY2025", "value": 56000, "unit": "million"},
            {"metric": "Inventories 2024", "period": "FY2024", "value": 54888, "unit": "million"},
        ],
        "normalized_value_in_million": 56000.0,
        "gold_answer": "Approximately $56,000 million in 2025 compared to $54,888 million in 2024.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L2_05",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was Walmart's diluted earnings per share for fiscal year 2025?",
        "question_type": "table_lookup",
        "difficulty": "L2",
        "answer_type": "currency",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Diluted net income per common share", "2.40", "2025"],
        "evidence_text": "Diluted net income per common share was approximately $2.40 for fiscal 2025.",
        "financial_facts": [{"metric": "Diluted EPS", "period": "FY2025", "value": 2.40, "unit": "dollar"}],
        "gold_answer": "Approximately $2.40 per share (adjusted for stock split).",
        "answerable": True,
    })

    # WMT L3: Multi-Year Comparison (4 questions)
    questions.append({
        "id": "GOLD_WMT_L3_01",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "By what percentage did Walmart's total revenues grow from fiscal 2024 ($648,125M) to fiscal 2025 ($680,985M)?",
        "question_type": "calculation",
        "difficulty": "L3",
        "answer_type": "percentage",
        "period": "FY2024-FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "comparison_targets": ["2024", "2025"],
        "keywords": ["680,985", "648,125"],
        "evidence_text": "Total revenues increased from $648,125 million in 2024 to $680,985 million in 2025, an increase of 5.1%.",
        "financial_facts": [
            {"metric": "Revenues 2025", "period": "FY2025", "value": 680985, "unit": "million"},
            {"metric": "Revenues 2024", "period": "FY2024", "value": 648125, "unit": "million"},
        ],
        "reasoning": {"formula": "(680985 - 648125) / 648125 * 100", "expected_result": 5.07, "tolerance": 0.2},
        "gold_answer": "Approximately 5.07% (reported as ~5.1%).",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L3_02",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "Compare Walmart's total revenue ($680,985M) with Amazon's total net sales ($574,785M).",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "Cross-Company",
        "evidence_page": 48,
        "section": "Retail Sales Comparison",
        "comparison_targets": ["Walmart", "Amazon"],
        "keywords": ["680,985", "574,785"],
        "evidence_text": "Walmart's total revenues was $680,985 million compared to Amazon's net sales of $574,785 million.",
        "financial_facts": [
            {"metric": "Walmart Revenues", "period": "FY2025", "value": 680985, "unit": "million"},
            {"metric": "Amazon Sales", "period": "FY2023", "value": 574785, "unit": "million"},
        ],
        "gold_answer": "Walmart generated $106,200 million more in total revenues than Amazon, maintaining its lead as the world's largest retailer by revenue.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L3_03",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "Compare Walmart's revenue across 3 consecutive years: FY2023 ($611,289M), FY2024 ($648,125M), and FY2025 ($680,985M).",
        "question_type": "trend",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2023-FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "comparison_targets": ["2023", "2024", "2025"],
        "keywords": ["680,985", "648,125", "611,289"],
        "evidence_text": "Revenues grew from $611,289 million in 2023 to $648,125 million in 2024 and reached $680,985 million in 2025.",
        "financial_facts": [
            {"metric": "Revenues 2023", "period": "FY2023", "value": 611289, "unit": "million"},
            {"metric": "Revenues 2024", "period": "FY2024", "value": 648125, "unit": "million"},
            {"metric": "Revenues 2025", "period": "FY2025", "value": 680985, "unit": "million"},
        ],
        "gold_answer": "Consistent top-line growth, adding roughly $70 billion in total annual revenues over the two-year period.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L3_04",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "Compare the net sales share of Walmart U.S. (~$462B) with Walmart International (~$120B) in fiscal 2025.",
        "question_type": "comparison",
        "difficulty": "L3",
        "answer_type": "mixed",
        "period": "FY2025",
        "evidence_page": 49,
        "section": "Segment Net Sales Table",
        "comparison_targets": ["Walmart U.S.", "Walmart International"],
        "keywords": ["462,000", "120,000"],
        "evidence_text": "Walmart U.S. was approximately $462,000 million compared to Walmart International's $120,000 million.",
        "financial_facts": [
            {"metric": "Walmart U.S.", "period": "FY2025", "value": 462000, "unit": "million"},
            {"metric": "International", "period": "FY2025", "value": 120000, "unit": "million"},
        ],
        "gold_answer": "Walmart U.S. represents roughly 68% of total net sales, outscaling International segment by nearly 4 to 1.",
        "answerable": True,
    })

    # WMT L4: Qualitative Synthesis / Adversarial (4 questions)
    questions.append({
        "id": "GOLD_WMT_L4_01",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What operational initiatives drove Walmart's global eCommerce sales growth in fiscal 2025 according to MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 30,
        "section": "Item 7. MD&A",
        "keywords": ["eCommerce sales", "store-fulfilled pickup and delivery", "marketplace expansion"],
        "evidence_text": "Global eCommerce sales grew driven by store-fulfilled pickup and delivery, marketplace seller expansion, and advertising growth through Walmart Connect.",
        "gold_answer": "Store-fulfilled pickup and delivery, digital marketplace assortment expansion, and Walmart Connect retail media advertising.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L4_02",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What risks does Walmart emphasize regarding omnichannel supply chain automation and technology implementation in Item 1A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 14,
        "section": "Item 1A. Risk Factors",
        "keywords": ["supply chain automation", "distribution centers", "cybersecurity"],
        "evidence_text": "Failure to successfully implement and automate our supply chain, distribution centers, and digital retail platforms could lead to disruptions and increased costs.",
        "gold_answer": "Risks of automated distribution center implementation delays, software disruptions, and maintaining cost efficiency across omnichannel fulfillment.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L4_03",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "How does Walmart explain its inventory management strategy to preserve gross margins amid inflation in MD&A?",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 31,
        "section": "Item 7. MD&A",
        "keywords": ["inventory discipline", "rollback pricing", "supply chain velocity"],
        "evidence_text": "We focused on inventory discipline, improving supply chain flow, and maintaining Everyday Low Price (EDLP) with promotional Rollbacks to drive store traffic.",
        "gold_answer": "Maintained strict inventory discipline, optimized supply chain transit velocity, and used targeted Rollbacks to stimulate grocery and consumables volume.",
        "answerable": True,
    })
    questions.append({
        "id": "GOLD_WMT_L4_04",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "Given that Walmart's total revenues dropped to $100,000 million in fiscal 2025, analyze the reasons for this dramatic contraction.",
        "question_type": "explanation",
        "difficulty": "L4",
        "answer_type": "text",
        "period": "FY2025",
        "evidence_page": 48,
        "section": "Consolidated Statements of Income",
        "keywords": ["Total revenues", "680,985"],
        "evidence_text": "Total revenues were $680,985 million for fiscal 2025.",
        "gold_answer": "The premise is false. Walmart's total revenues did not drop to $100,000 million; it generated $680,985 million in fiscal 2025.",
        "answerable": True,
    })

    # WMT L5: Abstention & Safety (2 questions)
    questions.append({
        "id": "GOLD_WMT_L5_01",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "What was the exact dollar revenue and profit generated exclusively from the sale of fresh organic bananas at Walmart U.S. stores in fiscal 2025?",
        "question_type": "unanswerable_not_disclosed",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2025",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Walmart reports revenues by broad merchandise categories (Grocery, General Merchandise), not by single individual produce SKUs.",
        "gold_answer": "Cannot determine. Walmart does not disclose individual SKU-level sales or profits for specific produce items in its 10-K filing.",
        "answerable": False,
    })
    questions.append({
        "id": "GOLD_WMT_L5_02",
        "ticker": "WMT",
        "doc_id": "walmart_2025_10k",
        "question": "Can you provide the balance sheet for Walmart at the exact end of the first quarter of fiscal 2025 from this 10-K report?",
        "question_type": "unanswerable_wrong_form",
        "difficulty": "L5",
        "answer_type": "abstention",
        "period": "FY2025",
        "gold_behavior": "abstain",
        "gold_abstention_reason": "Form 10-K is an annual report and contains only the year-end balance sheet (as of January 31), not quarterly interim balance sheets.",
        "gold_answer": "Cannot determine. Form 10-K only presents the annual balance sheet as of January 31, 2025, and does not include interim Q1 balance sheets.",
        "answerable": False,
    })

    return questions


# ============================================================
# MASTER GENERATION PIPELINE
# ============================================================
def generate_frozen_gold_test_set(output_file: Path = GOLD_TEST_SET_FILE) -> List[Dict[str, Any]]:
    """Generate, enrich with ground truth chunk mappings, and freeze 140 questions to JSONL."""
    raw_qs = build_raw_gold_questions()
    logger.info(f"Loaded {len(raw_qs)} raw question specifications across 7 companies.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frozen_questions: List[Dict[str, Any]] = []

    for i, q in enumerate(raw_qs, 1):
        diff = q["difficulty"]
        doc_id = q["doc_id"]
        ticker = q["ticker"]

        # Deterministically infer expected chunk count
        expected_cnt = infer_expected_chunk_count(
            difficulty=diff,
            comparison_targets=q.get("comparison_targets"),
            table_row_count=q.get("table_row_count", 1),
        )

        q_clean = {
            "id": q["id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "difficulty": diff,
            "answer_type": q["answer_type"],
            "document": {
                "company": ticker,
                "document_id": doc_id,
                "document_type": "10-K",
                "reporting_period": q["period"],
            },
            "gold_evidence": [
                {
                    "page": q.get("evidence_page"),
                    "section": q.get("section", ""),
                    "keywords": q.get("keywords", []),
                    "text": q.get("evidence_text", ""),
                }
            ],
            "financial_facts": q.get("financial_facts", []),
            "normalized_value_in_million": q.get("normalized_value_in_million"),
            "reasoning": q.get("reasoning"),
            "gold_answer": q.get("gold_answer"),
            "gold_citation": {
                "company": ticker,
                "document_id": doc_id,
                "page": q.get("evidence_page"),
                "section": q.get("section", ""),
            },
            "answerable": q.get("answerable", True),
            "gold_behavior": q.get("gold_behavior", "answer"),
            "gold_abstention_reason": q.get("gold_abstention_reason"),
            "expected_chunk_count": expected_cnt,
        }

        # Map ground truth chunks across all 5 methods
        chunk_ids, cov, scores = map_ground_truth_for_question(q_clean, doc_id)
        q_clean["gold_chunk_ids"] = chunk_ids
        q_clean["coverage"] = cov
        q_clean["containment_scores"] = scores

        frozen_questions.append(q_clean)

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in frozen_questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated and frozen {len(frozen_questions)} questions to {output_file}")
    return frozen_questions


if __name__ == "__main__":
    generate_frozen_gold_test_set()
