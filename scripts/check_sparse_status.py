"""Inspect Qdrant Collections to verify Sparse BM25 Vectors status.

Usage:
    python scripts/check_sparse_status.py
"""

import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC = ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from financial_rag.config import (
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    QDRANT_STORAGE_PATH,
    get_collection_name,
)


def inspect_collections():
    print(f"\n{'='*75}")
    print(f"[STATUS CHECK] KIEM TRA TRANG THAI SPARSE BM25 VECTORS TRONG QDRANT DB")
    print(f"Vi tri Qdrant Local: {QDRANT_STORAGE_PATH.resolve()}")
    print(f"{'='*75}\n")

    if not QDRANT_STORAGE_PATH.exists():
        print(f"[ERROR] Thu muc Qdrant khong ton tai: {QDRANT_STORAGE_PATH}")
        return

    client = QdrantClient(path=str(QDRANT_STORAGE_PATH))
    existing_collections = [c.name for c in client.get_collections().collections]

    print(f"Tong so collections trong Qdrant: {len(existing_collections)}")

    for method_key, method_info in CHUNK_METHODS.items():
        coll_name = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, method_key)
        disp_name = method_info["display_name"]

        print(f"\n--- [{disp_name}] ---")
        print(f"  * Collection Name: {coll_name}")

        if coll_name not in existing_collections:
            print(f"  [WARN] Collection chua duoc khoi tao!")
            continue

        info = client.get_collection(collection_name=coll_name)
        points_count = info.points_count
        print(f"  * Tong so Points: {points_count:,}")

        # Lay mau 1 point de kiem tra cau truc vector
        sample_points, _ = client.scroll(
            collection_name=coll_name,
            limit=1,
            with_vectors=True,
            with_payload=True,
        )

        if not sample_points:
            print(f"  [WARN] Collection rong, chua co chunk nao!")
            continue

        sample = sample_points[0]
        vectors = sample.vector

        has_dense = False
        has_sparse = False
        sparse_len = 0

        if isinstance(vectors, dict):
            has_dense = "dense" in vectors and vectors["dense"] is not None
            has_sparse = "sparse" in vectors and vectors["sparse"] is not None
            if has_sparse:
                sparse_obj = vectors["sparse"]
                if hasattr(sparse_obj, "indices"):
                    sparse_len = len(sparse_obj.indices)
                elif isinstance(sparse_obj, dict):
                    sparse_len = len(sparse_obj.get("indices", []))

        print(f"  * Sample Point ID: {sample.id}")
        print(f"  * Vector Dense (BGE-base 768-dim): {'[OK] DA NAP' if has_dense else '[FAIL] CHUA CO'}")
        
        if has_sparse and sparse_len > 0:
            print(f"  * Vector Sparse (BM25 Lexical):    [OK] DA NAP ({sparse_len} non-zero terms)")
        else:
            print(f"  * Vector Sparse (BM25 Lexical):    [CHUA NAP] (Sparse vector dang rong)")

    # Thu nghiem truy van test neu co it nhat 1 collection co sparse
    print(f"\n{'='*75}")
    print("Thu nghiem truy van Sparse BM25 tren Qdrant:")
    try:
        from fastembed import SparseTextEmbedding
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        test_q = "Apple total net sales 2024"
        s_res = list(sparse_model.embed([test_q]))[0]
        s_vec = rest.SparseVector(indices=s_res.indices.tolist(), values=s_res.values.tolist())

        test_coll = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, "method5_proposed_golden_hybrid")
        test_hits = client.query_points(
            collection_name=test_coll,
            query=s_vec,
            using="sparse",
            limit=3,
            with_payload=True,
        )
        print(f"  * Query: '{test_q}' tren '{test_coll}':")
        if test_hits.points:
            print(f"  [OK] Tim thay {len(test_hits.points)} chunks qua BM25:")
            for rank, pt in enumerate(test_hits.points, 1):
                p_load = pt.payload or {}
                print(f"     Rank {rank}: [score={pt.score:.4f}] {p_load.get('ticker')} | chunk_id={p_load.get('chunk_id')}")
        else:
            print(f"  [INFO] Chua co ket qua BM25 (Bo vector sparse chua duoc nap vao collection nay).")
    except Exception as e:
        print(f"  [WARN] Kiem tra truy van: {e}")

    print(f"{'='*75}\n")
    client.close()


if __name__ == "__main__":
    inspect_collections()
