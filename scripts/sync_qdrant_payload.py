"""Sync and Enrich Qdrant Payload In-Place (Zero Re-Embedding).

Enriches all 60,432 points across the 5 Qdrant collections with:
- Standardized 9 payload indexes.
- In-place 'period_years' (integer array) for multi-year financial statements.
- Verified 'document_id' (lowercase canonical document identifier).

Usage:
    python scripts/sync_qdrant_payload.py
"""

import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC = ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

from financial_rag.config import (
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    QDRANT_STORAGE_PATH,
    get_collection_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SyncQdrantPayload")


def sync_collections_payload():
    start_all = time.perf_counter()
    logger.info("=" * 70)
    logger.info("STARTING IN-PLACE QDRANT PAYLOAD SYNC (ZERO RE-EMBEDDING)")
    logger.info(f"Target Qdrant Path: {QDRANT_STORAGE_PATH.resolve()}")
    logger.info("=" * 70)

    if not QDRANT_STORAGE_PATH.exists():
        logger.error(f"Qdrant storage path does not exist: {QDRANT_STORAGE_PATH}")
        return

    client = QdrantClient(path=str(QDRANT_STORAGE_PATH))

    index_fields = [
        ("ticker", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("fiscal_year", PayloadSchemaType.INTEGER),
        ("period_years", PayloadSchemaType.INTEGER),
        ("fiscal_period", PayloadSchemaType.KEYWORD),
        ("statement_type", PayloadSchemaType.KEYWORD),
        ("form_type", PayloadSchemaType.KEYWORD),
        ("contains_table", PayloadSchemaType.BOOL),
        ("contains_text", PayloadSchemaType.BOOL),
    ]

    total_updated_points = 0

    for method_key, method_info in CHUNK_METHODS.items():
        coll_name = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, method_key)
        disp_name = method_info["display_name"]

        logger.info(f"\nProcessing [{disp_name}] -> Collection: {coll_name}...")

        try:
            coll_info = client.get_collection(coll_name)
        except Exception as e:
            logger.warning(f"Collection {coll_name} not found or error: {e}. Skipping.")
            continue

        points_count = coll_info.points_count
        logger.info(f"  * Total points: {points_count:,}")

        # 1. Register Payload Indexes
        for field_name, field_type in index_fields:
            try:
                client.create_payload_index(
                    collection_name=coll_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                logger.debug(f"Index creation note for {field_name}: {e}")

        # 2. Scroll and update payload in-place
        offset = None
        batch_size = 500
        coll_updated = 0

        while True:
            records, next_offset = client.scroll(
                collection_name=coll_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            # Group points by (period_years, document_id) for bulk set_payload calls
            update_groups: Dict[Tuple[Tuple[int, ...], str], List[str]] = defaultdict(list)

            for rec in records:
                payload = rec.payload or {}
                content = payload.get("content", "") or payload.get("content_retrieval", "")
                chunk_id = payload.get("chunk_id", str(rec.id))
                raw_fy = payload.get("fiscal_year")
                raw_doc_id = payload.get("document_id")
                raw_period_years = payload.get("period_years")

                # Infer document_id if missing
                if raw_doc_id:
                    doc_id = str(raw_doc_id).lower()
                else:
                    parts = str(chunk_id).lower().split("_")
                    if len(parts) >= 2 and re.match(r"^20\d\d$", parts[1]):
                        doc_id = f"{parts[0]}_{parts[1]}_10k"
                    else:
                        doc_id = ""

                # Compute period_years
                if raw_period_years and isinstance(raw_period_years, list) and len(raw_period_years) > 0:
                    period_years = sorted(list({int(y) for y in raw_period_years if str(y).isdigit()}))
                else:
                    found_years = {int(y) for y in re.findall(r"\b(20[12]\d)\b", content)}
                    if not found_years and raw_fy:
                        found_years = {int(raw_fy)}
                    elif not found_years and doc_id:
                        m_yr = re.search(r"\b(20\d\d)\b", doc_id)
                        if m_yr:
                            found_years = {int(m_yr.group(1))}
                    period_years = sorted(list(found_years)) if found_years else [2024]

                # Only queue update if period_years is missing or document_id needed fix
                needs_update = (
                    "period_years" not in payload
                    or payload.get("period_years") != period_years
                    or ("document_id" not in payload and doc_id)
                )

                if needs_update:
                    update_groups[(tuple(period_years), doc_id)].append(str(rec.id))

            # Apply batch payload updates
            for (p_years, d_id), point_ids in update_groups.items():
                patch: Dict[str, Any] = {"period_years": list(p_years)}
                if d_id:
                    patch["document_id"] = d_id
                client.set_payload(
                    collection_name=coll_name,
                    payload=patch,
                    points=point_ids,
                )
                coll_updated += len(point_ids)

            offset = next_offset
            if offset is None:
                break

        logger.info(f"  [OK] Successfully updated {coll_updated:,} points in '{coll_name}'.")
        total_updated_points += coll_updated

    elapsed = time.perf_counter() - start_all
    logger.info("\n" + "=" * 70)
    logger.info(f"[COMPLETED] In-place payload sync finished in {elapsed:.2f}s!")
    logger.info(f"Total points updated: {total_updated_points:,}")
    logger.info("=" * 70)

    # Clean close
    if hasattr(client, "close"):
        client.close()


if __name__ == "__main__":
    sync_collections_payload()
