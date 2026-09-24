"""Gold Test Set management and Ground Truth relevance verification (SSOT)."""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from financial_rag.config import GOLD_TEST_SET_FILE
from financial_rag.schema import (
    CONTAINMENT_THRESHOLDS,
    LEVEL_THRESHOLDS,
    compute_containment,
    extract_chunk_text,
    normalize_financial_number,
    normalize_numbers,
)

logger = logging.getLogger(__name__)


@dataclass
class GoldQuestion:
    id: str
    question: str
    question_type: str
    difficulty: str
    answer_type: str
    document: Dict[str, Any]
    gold_evidence: List[Dict[str, Any]]
    financial_facts: List[Dict[str, Any]]
    reasoning: Optional[Dict[str, Any]] = None
    gold_answer: Any = None
    gold_citation: Optional[Dict[str, Any]] = None
    answerable: bool = True
    gold_behavior: str = "answer"
    gold_chunk_ids: Dict[str, Any] = field(default_factory=dict)

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

            q_id = data.get("id") or data.get("query_id", "Q_UNKNOWN")
            q_text = data.get("question", "")
            q_type = data.get("question_type") or data.get("query_type", "factual")
            diff = data.get("difficulty", "L1")
            ans_type = data.get("answer_type", "text")

            doc = data.get("document") or {
                "company": data.get("target_company", ""),
                "document_id": data.get("target_company", "").lower() + "_10k_2025",
                "document_type": "10-K",
                "reporting_period": "FY2025",
            }

            ev = data.get("gold_evidence") or [{
                "page": data.get("target_page"),
                "section": data.get("target_statement", ""),
                "keywords": data.get("keywords", []),
            }]

            questions.append(GoldQuestion(
                id=q_id,
                question=q_text,
                question_type=q_type,
                difficulty=diff,
                answer_type=ans_type,
                document=doc,
                gold_evidence=ev,
                financial_facts=data.get("financial_facts", []),
                reasoning=data.get("reasoning"),
                gold_answer=data.get("gold_answer", data.get("expected_answer")),
                gold_citation=data.get("gold_citation"),
                answerable=data.get("answerable", True),
                gold_behavior=data.get("gold_behavior", "answer"),
                gold_chunk_ids=data.get("gold_chunk_ids", {}),
            ))

    logger.info(f"Loaded {len(questions)} gold evaluation questions from {file_path}")
    return questions


def save_gold_test_set(questions: List[GoldQuestion], file_path: Path = GOLD_TEST_SET_FILE) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(questions)} gold questions to {file_path}")


TICKER_MAP = {
    "APPLE": "AAPL",
    "AMAZON": "AMZN",
    "INTEL": "INTC",
    "NIKE": "NKE",
    "NVIDIA": "NVDA",
    "WALMART": "WMT",
    "ADVANCED MICRO DEVICES": "AMD",
}


def normalize_chunk_id(cid: str) -> str:
    cid = str(cid).lower().strip()
    cid = cid.replace('_sec_', '_btag_').replace('_narr_', '_nar_')
    cid = re.sub(r'_c0*(\d+)', r'_c\1', cid)
    cid = re.sub(r'_tbl_0*(\d+)', r'_tbl_\1', cid)
    cid = re.sub(r'_nar_0*(\d+)', r'_nar_\1', cid)
    cid = re.sub(r'_btag_0*(\d+)', r'_btag_\1', cid)
    return cid


def _gold_get(gold_q: Any, key: str, default: Any = None) -> Any:
    """Uniform accessor for both dataclass and dict gold questions."""
    if isinstance(gold_q, dict):
        return gold_q.get(key, default)
    return getattr(gold_q, key, default)


def _check_answerable(gold_q: Any) -> bool:
    answerable = _gold_get(gold_q, "answerable", True)
    if not answerable:
        return False
    behavior = _gold_get(gold_q, "gold_behavior", "answer")
    if behavior == "abstain":
        return False
    return True


def _get_target_ids(gold_q: Any, method_key: Optional[str]) -> List[str]:
    """Return normalized explicit chunk IDs for the method, flattening dict/list shapes."""
    gold_chunks = _gold_get(gold_q, "gold_chunk_ids", {}) or {}
    explicit = gold_chunks.get(method_key, []) if method_key else []
    if isinstance(explicit, dict):
        ids = []
        for k, v in explicit.items():
            if isinstance(v, list):
                ids.extend(v)
            elif isinstance(v, str) and v:
                ids.append(v)
        return ids
    if isinstance(explicit, list):
        return list(explicit)
    if isinstance(explicit, str) and explicit:
        return [explicit]
    return []


def _get_gold_keywords(gold_q: Any) -> List[str]:
    """Safely extract keyword list from both GoldQuestion dataclass and raw dict."""
    kws: List[str] = []
    if isinstance(gold_q, dict):
        if gold_q.get("keywords"):
            kws.extend(gold_q["keywords"])
        for ev in gold_q.get("gold_evidence", []):
            if isinstance(ev, dict) and ev.get("keywords"):
                kws.extend(ev["keywords"])
    else:
        raw_kws = getattr(gold_q, "keywords", [])
        if raw_kws:
            kws.extend(raw_kws)
        else:
            evs = getattr(gold_q, "gold_evidence", [])
            for ev in evs:
                if isinstance(ev, dict) and ev.get("keywords"):
                    kws.extend(ev["keywords"])
    return list(dict.fromkeys(kws))


def _check_id_match(
    chunk_payload: Dict[str, Any],
    gold_q: Any,
    method_key: Optional[str],
) -> Tuple[bool, bool]:
    """Returns (id_matched, has_content_signal).
    
    Even if ID matches, require minimum content signal to prevent
    TOC/heading contamination from being accepted via GT ID alone.
    """
    chunk_id = chunk_payload.get("chunk_id", "")
    target_ids = _get_target_ids(gold_q, method_key)
    if not target_ids:
        return False, False

    norm_target = {normalize_chunk_id(x) for x in target_ids}
    id_matched = normalize_chunk_id(chunk_id) in norm_target or chunk_id in target_ids
    if not id_matched:
        return False, False

    return True, _has_min_content_signal(chunk_payload, gold_q)


def _has_min_content_signal(chunk_payload: Dict[str, Any], gold_q: Any) -> bool:
    """Chunk has at least one gold keyword OR one gold number."""
    content = extract_chunk_text(chunk_payload).lower()
    content_norm = normalize_numbers(content)

    # FIX [B3]: Robust keyword extraction across dataclass and raw dict
    kws = _get_gold_keywords(gold_q)
    if kws:
        if any(str(kw).lower() in content for kw in kws):
            return True

    facts = _gold_get(gold_q, "financial_facts", [])
    for fact in facts:
        val = normalize_financial_number(str(fact.get("value", "")), strip_magnitude_suffix=True)
        if val:
            v = val.lstrip("-")
            if v and v in content_norm:
                return True
    return False


def _check_ticker(chunk_payload: Dict[str, Any], gold_q: Any) -> bool:
    chunk_ticker = str(chunk_payload.get("ticker", "")).upper()
    target_ticker = _gold_get(gold_q, "target_company", "") or ""
    if isinstance(gold_q, dict) and not target_ticker:
        target_ticker = gold_q.get("target_company") or gold_q.get("document", {}).get("company", "")
    target_ticker = TICKER_MAP.get(target_ticker.lower(), target_ticker).upper()

    if target_ticker and target_ticker != "MULTI" and chunk_ticker and chunk_ticker != target_ticker:
        return False
    return True


def _extract_question_years(question: str) -> List[int]:
    return [int(y) for y in re.findall(r"(20\d{2})", question)]


def _check_fiscal_year(
    chunk_payload: Dict[str, Any],
    gold_q: Any,
) -> bool:
    """Level-aware fiscal year precision."""
    level = _gold_get(gold_q, "difficulty", "L1")
    if level == "L4":
        return True  # Qualitative disclosures don't strictly require fiscal year filtering

    doc_info = _gold_get(gold_q, "document", {}) or {}
    doc_id = str(doc_info.get("document_id", "")).lower()
    chunk_doc_id = str(chunk_payload.get("document_id") or chunk_payload.get("doc_id") or "").lower()
    same_doc = bool(doc_id and chunk_doc_id and (doc_id in chunk_doc_id or chunk_doc_id in doc_id))

    # FIX [B4]: If from the exact same filing document (e.g. apple_2024_10k), accept immediately
    if same_doc:
        return True

    # Extract valid comparison / reporting years from question document
    doc_years: set = set()
    period_str = str(doc_info.get("reporting_period", ""))
    for y in re.findall(r"(20\d{2})", period_str):
        doc_years.add(int(y))
    if "comparison_years" in doc_info:
        doc_years.update(int(y) for y in doc_info["comparison_years"] if str(y).isdigit())
    if not doc_years:
        doc_years = set(_extract_question_years(_gold_get(gold_q, "question", "")))

    chunk_period_years = chunk_payload.get("period_years") or []
    chunk_years_set = {int(y) for y in chunk_period_years if str(y).isdigit()}
    chunk_filing_year = chunk_payload.get("fiscal_year")
    if chunk_filing_year and not chunk_years_set:
        chunk_years_set = {int(chunk_filing_year)}

    # FIX [B4]: If both sides provide year metadata, verify non-empty intersection
    if doc_years and chunk_years_set:
        return bool(doc_years & chunk_years_set)

    return True


def _check_keywords(chunk_payload: Dict[str, Any], gold_q: Any) -> bool:
    """Level-aware keyword ratio threshold."""
    level = _gold_get(gold_q, "difficulty", "L1")
    threshold = LEVEL_THRESHOLDS.get(level)
    min_ratio = threshold.keywords_required if threshold else 0.60

    kws = _get_gold_keywords(gold_q)
    if not kws:
        return True  # no keywords → pass

    content = extract_chunk_text(chunk_payload)
    content_lower = content.lower()
    content_norm = normalize_numbers(content_lower)

    matched = 0
    for kw in kws:
        kw_str = str(kw).lower()
        kw_norm = normalize_financial_number(kw_str, strip_magnitude_suffix=True).lower()
        if kw_str in content_lower or (kw_norm and kw_norm in content_norm):
            matched += 1

    return (matched / len(kws)) >= min_ratio


def _check_page(chunk_payload: Dict[str, Any], gold_q: Any) -> bool:
    target_pg = _gold_get(gold_q, "target_page", None)
    if target_pg is None and isinstance(gold_q, dict):
        evs = gold_q.get("gold_evidence", [])
        if evs and evs[0].get("page"):
            target_pg = evs[0].get("page")
    if target_pg is None:
        return True

    chunk_pages = [int(p) for p in chunk_payload.get("source_pages", []) if str(p).isdigit()]
    return int(target_pg) in chunk_pages


def is_chunk_relevant(
    chunk_payload: Dict[str, Any],
    gold_q: Any,
    method_key: Optional[str] = None,
) -> bool:
    """Determine whether a retrieved chunk is a true positive for the gold question.

    Order of checks:
    1. Answerable + non-abstain
    2. Explicit ID match (with content signal validation)
    3. Ticker match
    4. Level-aware fiscal year match
    5. Level-aware keyword ratio
    6. Page correlation
    """
    if not _check_answerable(gold_q):
        return False

    id_matched, has_signal = _check_id_match(chunk_payload, gold_q, method_key)
    if id_matched:
        if has_signal:
            return True
        logger.warning(
            f"Chunk {chunk_payload.get('chunk_id')} matched ID but failed content signal — "
            f"falling through to keyword check for question {_gold_get(gold_q, 'id', '?')}"
        )

    if not _check_ticker(chunk_payload, gold_q):
        return False
    if not _check_fiscal_year(chunk_payload, gold_q):
        return False
    if not _check_keywords(chunk_payload, gold_q):
        return False
    if not _check_page(chunk_payload, gold_q):
        return False

    return True


def text_pool_satisfies_signature(gold_q: Any, pool_text: str) -> bool:
    """Check if union of top-k chunk texts contains enough evidence.

    Level-aware thresholds from LEVEL_THRESHOLDS:
    - L1/L2: 100% numbers + 60% keywords
    - L3:    75% numbers + 60% keywords
    - L4:    0% numbers (qualitative) + 60% keywords
    - L5:    not applicable
    """
    level = _gold_get(gold_q, "difficulty", "L1")
    if level == "L5":
        return False

    threshold = LEVEL_THRESHOLDS.get(level)
    if threshold is None:
        return False

    pool_lower = pool_text.lower()
    pool_norm = normalize_numbers(pool_lower).replace(",", "")

    # --- Numbers check ---
    facts = _gold_get(gold_q, "financial_facts", [])
    gold_numbers = set()
    for fact in facts:
        raw = str(fact.get("value", "")).strip()
        if not raw:
            continue
        norm = normalize_financial_number(raw, strip_magnitude_suffix=True)
        if norm:
            gold_numbers.add(norm.replace(",", "").lstrip("-"))

    if gold_numbers and threshold.numbers_required > 0:
        matched = sum(1 for n in gold_numbers if n in pool_norm)
        if matched / len(gold_numbers) < threshold.numbers_required:
            return False

    # --- Keywords check ---
    kws = _gold_get(gold_q, "keywords", [])
    if isinstance(gold_q, dict) and not kws:
        for ev in gold_q.get("gold_evidence", []):
            kws.extend(ev.get("keywords", []))

    if kws and threshold.keywords_required > 0:
        matched_kw = sum(1 for kw in kws if str(kw).lower() in pool_lower)
        if matched_kw / len(kws) < threshold.keywords_required:
            return False

    return True
