"""Gold Test Set management and Ground Truth relevance verification.

Conforms to the multi-tier Ground Truth specification:
Retrieval Evidence -> Financial Facts -> Reasoning/Calculation -> Gold Answer -> Citation.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from script.financial_rag.config import GOLD_TEST_SET_FILE

logger = logging.getLogger(__name__)


@dataclass
class GoldQuestion:
    """A comprehensive ground truth evaluation question for multi-tier RAG benchmark."""
    id: str
    question: str
    question_type: str                     # factual, table_lookup, calculation, comparison, multi_hop, explanation, trend, unanswerable
    difficulty: str                        # L1 (direct), L2 (table reasoning), L3 (single calc), L4 (multi-step calc), L5 (multi-hop / abstention)
    answer_type: str                       # currency, percentage, number, text, list, boolean, abstention, mixed
    document: Dict[str, Any]               # company, document_id, document_type, reporting_period
    gold_evidence: List[Dict[str, Any]]    # page, section, table_id, chunk_id, keywords
    financial_facts: List[Dict[str, Any]]  # metric, period, value, unit
    reasoning: Optional[Dict[str, Any]] = None  # inputs, formula, expected_result, tolerance
    gold_answer: Any = None
    gold_citation: Optional[Dict[str, Any]] = None
    answerable: bool = True
    gold_behavior: str = "answer"          # answer, abstain, flag_conflict
    gold_chunk_ids: Dict[str, List[str]] = field(default_factory=dict)  # method_key -> [chunk_id]

    # Backward compatibility properties
    @property
    def query_id(self) -> str:
        return self.id

    @property
    def target_company(self) -> str:
        return str(self.document.get("company", "")).upper()

    @property
    def query_type(self) -> str:
        return self.question_type

    @property
    def expected_answer(self) -> str:
        return str(self.gold_answer) if self.gold_answer is not None else ""

    @property
    def target_page(self) -> Optional[int]:
        for ev in self.gold_evidence:
            if "page" in ev and ev["page"] is not None:
                return int(ev["page"])
        return None

    @property
    def keywords(self) -> List[str]:
        kws = []
        for ev in self.gold_evidence:
            for kw in ev.get("keywords", []):
                if kw not in kws:
                    kws.append(kw)
        if not kws:
            for fact in self.financial_facts:
                val = str(fact.get("value", ""))
                metric = str(fact.get("metric", ""))
                if val and val not in kws:
                    kws.append(val)
                if metric and metric not in kws:
                    kws.append(metric)
        return kws


def load_gold_test_set(file_path: Path = GOLD_TEST_SET_FILE) -> List[GoldQuestion]:
    """Load gold evaluation questions from JSONL file supporting full schema and legacy format."""
    if not file_path.exists():
        logger.warning(f"Gold test set file does not exist: {file_path}")
        return []

    questions: List[GoldQuestion] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)

            # Adapt from legacy or minimal format if needed
            q_id = data.get("id") or data.get("query_id", "Q_UNKNOWN")
            q_text = data.get("question", "")
            q_type = data.get("question_type") or data.get("query_type", "factual")
            diff = data.get("difficulty", "L1")
            ans_type = data.get("answer_type", "text")

            doc = data.get("document")
            if not doc:
                doc = {
                    "company": data.get("target_company", ""),
                    "document_id": data.get("target_company", "").lower() + "_10k_2025",
                    "document_type": "10-K",
                    "reporting_period": "FY2025",
                }

            ev = data.get("gold_evidence")
            if not ev:
                ev = [
                    {
                        "page": data.get("target_page"),
                        "section": data.get("target_statement", ""),
                        "keywords": data.get("keywords", []),
                    }
                ]

            facts = data.get("financial_facts", [])
            reasoning = data.get("reasoning")
            gold_ans = data.get("gold_answer", data.get("expected_answer"))
            citation = data.get("gold_citation")
            answerable = data.get("answerable", True)
            behavior = data.get("gold_behavior", "answer")
            chunk_ids = data.get("gold_chunk_ids", {})

            questions.append(
                GoldQuestion(
                    id=q_id,
                    question=q_text,
                    question_type=q_type,
                    difficulty=diff,
                    answer_type=ans_type,
                    document=doc,
                    gold_evidence=ev,
                    financial_facts=facts,
                    reasoning=reasoning,
                    gold_answer=gold_ans,
                    gold_citation=citation,
                    answerable=answerable,
                    gold_behavior=behavior,
                    gold_chunk_ids=chunk_ids,
                )
            )

    logger.info(f"Loaded {len(questions)} gold evaluation questions from {file_path}")
    return questions


def save_gold_test_set(questions: List[GoldQuestion], file_path: Path = GOLD_TEST_SET_FILE) -> None:
    """Save gold evaluation questions to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for q in questions:
            data = asdict(q)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(questions)} gold questions to {file_path}")


def is_chunk_relevant(
    chunk_payload: Dict[str, Any],
    gold_q: GoldQuestion,
    method_key: str,
) -> bool:
    """Determine whether a retrieved chunk is a true positive for the gold question.
    
    Cross-method evaluation strategy:
    1. Unanswerable / Abstention: Question cannot be answered by the corpus -> returns False.
    2. Direct Chunk ID match: If gold_q has explicit chunk_ids listed for this method.
    3. Company ticker match: Chunk company must match target company (unless MULTI).
    4. Signature Content match: All required keywords/numbers from gold_evidence or facts
       must appear in chunk's content or retrieval text.
    5. Page correlation: If specified in gold_evidence, matches source_pages.
    """
    if not gold_q.answerable or gold_q.gold_behavior == "abstain":
        return False

    # 1. Direct ID match
    explicit_ids = gold_q.gold_chunk_ids.get(method_key, [])
    if explicit_ids and chunk_payload.get("chunk_id") in explicit_ids:
        return True

    # 2. Company ticker match
    chunk_ticker = str(chunk_payload.get("ticker", "")).upper()
    target_ticker = gold_q.target_company
    if target_ticker != "MULTI" and chunk_ticker != target_ticker:
        return False

    # 3. Content signature match
    content = str(chunk_payload.get("content", "")) + " " + str(chunk_payload.get("content_retrieval", ""))
    content_lower = content.lower()

    kws = gold_q.keywords
    if not kws:
        return False

    all_matched = True
    for kw in kws:
        if kw.lower() not in content_lower:
            all_matched = False
            break

    if all_matched:
        target_pg = gold_q.target_page
        if target_pg:
            chunk_pages = chunk_payload.get("source_pages", [])
            if chunk_pages and target_pg in chunk_pages:
                return True
        return True

    return False
