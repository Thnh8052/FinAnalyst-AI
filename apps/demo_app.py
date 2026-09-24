import io
import json
import os
import sys
import time
import gc
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import streamlit as st

# Setup sys.path so financial_rag modules are always importable
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from financial_rag.config import (
    CHUNK_METHODS,
    DEFAULT_EMBEDDING_PROVIDER,
    OUTPUT_RETRIEVAL_ROOT,
    get_collection_name,
)
from financial_rag.retrieval import HybridRetriever, DenseRetriever
from financial_rag.testbed import load_gold_test_set, is_chunk_relevant

# ==============================================================================
# Page Configuration & Custom Styling
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="FinAnalyst-AI Studio: SEC 10-K & Retrieval Arena",
    page_icon="🏛️",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-red { background-color: #FEE2E2; color: #991B1B; }
    .badge-purple { background-color: #F3E8FF; color: #6B21A8; }
    .gold-box {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .retrieval-hit-box {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
    }
    .retrieval-miss-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
    }
    .chunk-card {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.7rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .hit-card {
        border: 2px solid #22C55E !important;
        background-color: #F0FDF4 !important;
    }
</style>
""", unsafe_allow_html=True)

SEC_FILINGS_DIR = ROOT_DIR / "data" / "sec_filings"
PARSING_DIR = ROOT_DIR / "outputs" / "parsing"
CHUNKING_DIR = ROOT_DIR / "outputs" / "chunking"
GOLD_FILE = ROOT_DIR / "data" / "gold_test_set" / "v0_gold_questions.jsonl"
SUMMARY_FILE = OUTPUT_RETRIEVAL_ROOT / "v0_retrieval_evaluation_summary.json"

TICKER_MAP = {
    "amazon": "AMZN",
    "amd": "AMD",
    "apple": "AAPL",
    "intel": "INTC",
    "nike": "NKE",
    "nvidia": "NVDA",
    "walmart": "WMT"
}

METHOD_NAMES = {
    "baseline_fixed_size": "Method 1: Naive Fixed-Size (512 tokens)",
    "baseline_deterministic_structure": "Method 2: Deterministic Layout-Aware",
    "method3_heading_llm": "Method 3: Semantic Heading + LLM Merge",
    "method4_boundary_tagging": "Method 4: LLM Boundary Tagging",
    "method5_proposed_golden_hybrid": "Method 5: Proposed Golden Hybrid (Tuples + Tables)"
}

FOLDER_TO_RETRIEVAL_KEY = {
    "baseline_fixed_size": "method1_fixed_size",
    "baseline_deterministic_structure": "method2_deterministic",
    "method3_heading_llm": "method3_heading_llm",
    "method4_boundary_tagging": "method4_boundary_tagging",
    "method5_proposed_golden_hybrid": "method5_proposed_golden_hybrid",
}

RETRIEVAL_METHODS = [
    {
        "key": "method1_fixed_size",
        "display": "Method 1: Fixed-Size",
        "badge_class": "badge-red",
        "badge_text": "⚠️ VỠ BẢNG",
        "desc": "Cắt thô 512 tokens, mất hàng/cột bảng"
    },
    {
        "key": "method2_deterministic",
        "display": "Method 2: Deterministic",
        "badge_class": "badge-amber",
        "badge_text": "❌ THIẾU TUPLES",
        "desc": "Markdown thô nhưng thiếu bộ ba Fact Tuples"
    },
    {
        "key": "method3_heading_llm",
        "display": "Method 3: Heading + LLM",
        "badge_class": "badge-blue",
        "badge_text": "🟡 GOM ĐỀ MỤC",
        "desc": "Gom nhóm theo heading cấu trúc"
    },
    {
        "key": "method4_boundary_tagging",
        "display": "Method 4: Boundary Tagging",
        "badge_class": "badge-purple",
        "badge_text": "🟡 THẺ BIÊN",
        "desc": "LLM gán thẻ phân đoạn ranh giới"
    },
    {
        "key": "method5_proposed_golden_hybrid",
        "display": "Method 5: Proposed Golden Hybrid",
        "badge_class": "badge-green",
        "badge_text": "✅ BẢO TOÀN BẢNG + TUPLES",
        "desc": "Bảng nguyên vẹn + Biểu diễn kép Fact Tuples"
    },
]

# ==============================================================================
# Cached Data Loaders
# ==============================================================================
@st.cache_data
def load_all_catalog():
    """Scans and catalogs all 35 available SEC 10-K filings."""
    catalog = []
    if not PARSING_DIR.exists():
        return catalog

    for doc_dir in sorted(PARSING_DIR.glob("*_10k")):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        parts = doc_id.split("_")
        company = parts[0].lower()
        year = parts[1] if len(parts) > 1 else ""
        ticker = TICKER_MAP.get(company, company.upper())

        pdf_path = SEC_FILINGS_DIR / company / f"{year}.pdf"
        if not pdf_path.exists():
            matches = list((SEC_FILINGS_DIR / company).glob("*.pdf")) if (SEC_FILINGS_DIR / company).exists() else []
            for m in matches:
                if year in m.name:
                    pdf_path = m
                    break

        page_count = 0
        pages_dir = doc_dir / "pages"
        if pages_dir.exists():
            page_count = len(list(pages_dir.glob("page_*.md")))

        if page_count == 0 and pdf_path.exists():
            try:
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0

        catalog.append({
            "doc_id": doc_id,
            "company": company.title(),
            "ticker": ticker,
            "year": year,
            "pdf_path": str(pdf_path),
            "parsing_dir": str(doc_dir),
            "page_count": max(page_count, 1),
            "display_name": f"{ticker} ({company.title()}) FY{year}"
        })
    return catalog


@st.cache_data
def load_gold_data():
    """Loads all benchmark questions and groups them by document and page."""
    gold_by_doc_page = defaultdict(list)
    gold_by_doc = defaultdict(list)
    all_questions = []

    if not GOLD_FILE.exists():
        return gold_by_doc_page, gold_by_doc, all_questions

    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
                all_questions.append(q)
                doc_info = q.get("document", {})
                company = doc_info.get("company", "").lower()
                period = doc_info.get("reporting_period", "")
                year = period.replace("FY", "").strip()

                doc_id_match = doc_info.get("document_id")
                if not doc_id_match:
                    for comp_key, tick in TICKER_MAP.items():
                        if tick.lower() == company.lower() or comp_key == company.lower():
                            doc_id_match = f"{comp_key}_{year}_10k"
                            break

                pages = []
                for ev in q.get("gold_evidence", []):
                    p = ev.get("page")
                    if p is not None:
                        pages.append(int(p))

                if doc_id_match:
                    gold_by_doc[doc_id_match].append(q)
                    for p in pages:
                        gold_by_doc_page[(doc_id_match, p)].append(q)
            except Exception:
                continue

    return gold_by_doc_page, gold_by_doc, all_questions


@st.cache_data
def load_chunks_for_method(doc_id: str, method_folder: str):
    """Loads chunks from outputs/chunking/<doc_id>/<method_folder>/chunks.jsonl."""
    chunk_file = CHUNKING_DIR / doc_id / method_folder / "chunks.jsonl"
    chunks = []
    if chunk_file.exists():
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except Exception:
                        continue
    return chunks


@st.cache_resource
def get_dense_retriever():
    """Instantiate and cache HybridRetriever singleton for high-speed multi-query inference."""
    return HybridRetriever()


@st.cache_data
def load_benchmark_summary():
    """Loads precomputed evaluation benchmark results."""
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_pdf_page_image(pdf_path: str, page_number: int, dpi: int = 150):
    """Renders a PDF page to PIL Image using PyMuPDF."""
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            doc.close()
            return None
        page = doc[page_number - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    except Exception:
        return None


def extract_question_pages(q):
    pages = []
    for ev in q.get("gold_evidence", []):
        p = ev.get("page")
        if p is not None:
            pages.append(int(p))
    return sorted(list(set(pages)))


catalog = load_all_catalog()
gold_by_doc_page, gold_by_doc, all_gold_questions = load_gold_data()
benchmark_summary = load_benchmark_summary()

if not catalog:
    st.error("Không tìm thấy dữ liệu báo cáo trong `outputs/parsing/` hoặc `data/sec_filings/`.")
    st.stop()

# ==============================================================================
# Sidebar: Mode Selection & Navigation
# ==============================================================================
st.sidebar.markdown("## 🧭 Không Gian Làm Việc")
app_mode = st.sidebar.radio(
    "Chọn Module Ứng Dụng:",
    [
        "📑 1. Kiểm Tra SEC 10-K & Chunking",
        "🔍 2. Retrieval Studio (Đấu Trường 5 Phương Pháp)"
    ],
    index=0,
    key="main_app_mode"
)
st.sidebar.markdown("---")

# ==============================================================================
# MODE 1: SEC 10-K & CHUNKING INSPECTOR (PRESERVED)
# ==============================================================================
if app_mode == "📑 1. Kiểm Tra SEC 10-K & Chunking":
    st.sidebar.markdown("## 📚 Thư Viện 35 Hồ Sơ SEC 10-K")

    # Filter by Ticker
    tickers = ["ALL"] + sorted(list(set(d["ticker"] for d in catalog)))
    selected_ticker = st.sidebar.selectbox("Lọc theo Tập đoàn (Ticker):", tickers, index=0)

    filtered_catalog = [d for d in catalog if selected_ticker == "ALL" or d["ticker"] == selected_ticker]

    # Doc Selection List
    doc_options = {}
    for d in filtered_catalog:
        gold_count = len(gold_by_doc.get(d["doc_id"], []))
        star = f" ⭐ [{gold_count} Gold Qs]" if gold_count > 0 else ""
        label = f"{d['display_name']} ({d['page_count']} trang){star}"
        doc_options[label] = d

    selected_label = st.sidebar.selectbox(
        "Chọn Báo Cáo Tài Chính 10-K:",
        list(doc_options.keys()),
        index=0
    )
    current_doc = doc_options[selected_label]

    # Manage Page Number in session_state
    if "current_doc_id" not in st.session_state or st.session_state["current_doc_id"] != current_doc["doc_id"]:
        st.session_state["current_doc_id"] = current_doc["doc_id"]
        st.session_state["current_page"] = 1

    max_pages = current_doc["page_count"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Điều Hướng Trang")

    # Quick Navigation Buttons (+ / - / First / Last)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.sidebar.columns(4)

    with nav_col1:
        if st.button("⏮ Đầu", use_container_width=True, help="Về trang 1"):
            st.session_state["current_page"] = 1
            st.rerun()

    with nav_col2:
        if st.button("◀ Trước", use_container_width=True, help="Lùi 1 trang"):
            if st.session_state["current_page"] > 1:
                st.session_state["current_page"] -= 1
                st.rerun()

    with nav_col3:
        if st.button("Tiếp ▶", use_container_width=True, help="Tiến 1 trang"):
            if st.session_state["current_page"] < max_pages:
                st.session_state["current_page"] += 1
                st.rerun()

    with nav_col4:
        if st.button("Cuối ⏭", use_container_width=True, help="Tới trang cuối"):
            st.session_state["current_page"] = max_pages
            st.rerun()

    # Direct numeric input
    page_num_input = st.sidebar.number_input(
        f"Nhập số trang (1 - {max_pages}):",
        min_value=1,
        max_value=max_pages,
        value=min(st.session_state["current_page"], max_pages),
        step=1,
        key="page_input"
    )
    if page_num_input != st.session_state["current_page"]:
        st.session_state["current_page"] = page_num_input
        st.rerun()

    # Slider for fast scrubbing
    page_slider = st.sidebar.slider(
        "Thanh kéo lướt trang:",
        min_value=1,
        max_value=max_pages,
        value=st.session_state["current_page"],
        key="page_slider"
    )
    if page_slider != st.session_state["current_page"]:
        st.session_state["current_page"] = page_slider
        st.rerun()

    # Quick Jump to Gold Question Pages
    doc_gold_list = gold_by_doc.get(current_doc["doc_id"], [])
    if doc_gold_list:
        gold_pages = sorted(list(set(
            p for q in doc_gold_list 
            for p in extract_question_pages(q)
        )))
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"⭐ **Các Trang Có Câu Hỏi Gold ({len(doc_gold_list)} câu):**")

        jump_options = ["-- Chọn trang để nhảy nhanh --"] + [
            f"Trang {p} ({len(gold_by_doc_page.get((current_doc['doc_id'], p), []))} câu hỏi)"
            for p in gold_pages
        ]
        selected_jump = st.sidebar.selectbox("Nhảy nhanh đến trang có Benchmark:", jump_options)
        if selected_jump != "-- Chọn trang để nhảy nhanh --":
            target_p = int(selected_jump.split()[1])
            if target_p != st.session_state["current_page"]:
                st.session_state["current_page"] = target_p
                st.rerun()

    # DPI setting for PDF rendering
    dpi_setting = st.sidebar.select_slider("Độ nét xem PDF (DPI):", options=[100, 150, 200], value=150)

    # Header
    current_page = st.session_state["current_page"]
    st.markdown(f"<div class='main-header'>🏛️ FinAnalyst-AI Studio: Đối Soát Parsing, Chunking & Ground Truth</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sub-header'>Hồ sơ: <b>{current_doc['display_name']}</b> | Mã tài liệu: <code>{current_doc['doc_id']}</code> | "
        f"Trang đang xem: <b style='color:#2563EB;'>Trang {current_page} / {max_pages}</b></div>",
        unsafe_allow_html=True
    )

    # 2 Columns (Left: PDF Original, Right: Parsing & Chunking Tabs)
    col_pdf, col_tabs = st.columns([1, 1], gap="medium")

    with col_pdf:
        st.subheader(f"📑 Bản Gốc SEC Form 10-K (Trang {current_page})")
        pdf_img = render_pdf_page_image(current_doc["pdf_path"], current_page, dpi=dpi_setting)
        if pdf_img is not None:
            st.image(pdf_img, caption=f"{current_doc['doc_id']} - Trang {current_page}", use_container_width=True)
        else:
            st.warning(f"Không thể mở file PDF gốc: `{current_doc['pdf_path']}`")

    with col_tabs:
        st.subheader("🔬 Không Gian Kiểm Tra Dữ Liệu")

        main_tab1, main_tab2, main_tab3 = st.tabs([
            "📄 1. Kết Quả Parsing (Trang này)",
            "🧩 2. Kết Quả Chunking (5 Phương pháp)",
            "⚖️ 3. So Sánh 5 Phương Pháp Tại Trang"
        ])

        with main_tab1:
            md_file = Path(current_doc["parsing_dir"]) / "pages" / f"page_{current_page:03d}.md"
            json_file = Path(current_doc["parsing_dir"]) / "pages" / f"page_{current_page:03d}.json"
            manifest_file = Path(current_doc["parsing_dir"]) / "routing_manifest.json"

            parse_sub1, parse_sub2, parse_sub3, parse_sub4 = st.tabs([
                "📖 Markdown Hiển Thị",
                "📝 Raw Markdown",
                "💻 Canonical JSON (AST)",
                "🔍 Profiler & Audit Routing"
            ])

            with parse_sub1:
                if md_file.exists():
                    md_content = md_file.read_text(encoding="utf-8")
                    st.markdown(md_content)
                else:
                    st.info(f"Chưa có file markdown parsed cho trang {current_page} (`{md_file.name}`).")

            with parse_sub2:
                if md_file.exists():
                    st.code(md_file.read_text(encoding="utf-8"), language="markdown")
                else:
                    st.info("Không có dữ liệu Markdown thô.")

            with parse_sub3:
                if json_file.exists():
                    try:
                        ast_data = json.loads(json_file.read_text(encoding="utf-8"))
                        st.json(ast_data)
                    except Exception as e:
                        st.error(f"Lỗi đọc JSON: {e}")
                else:
                    st.info(f"Không tìm thấy file Canonical JSON cho trang {current_page}.")

            with parse_sub4:
                if manifest_file.exists():
                    try:
                        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                        profile = next((p for p in manifest_data.get("profiles", []) if p.get("pdf_page") == current_page), None)
                        if profile:
                            st.markdown("**Kết quả phân loại tự động của Document Profiler:**")
                            col_p1, col_p2, col_p3 = st.columns(3)
                            col_p1.metric("Layout Class", profile.get("layout_class", "N/A"))
                            col_p2.metric("Table Score", f"{profile.get('table_density_score', 0):.2f}")
                            col_p3.metric("Financial Score", f"{profile.get('financial_term_score', 0):.2f}")
                            st.json(profile)
                        else:
                            st.info(f"Không tìm thấy thông tin Profile của trang {current_page} trong manifest.")
                    except Exception as e:
                        st.error(f"Lỗi đọc manifest: {e}")
                else:
                    st.info("Chưa có file routing_manifest.json.")

        with main_tab2:
            selected_method_key = st.radio(
                "Chọn Phương Pháp Phân Đoạn (Chunking Method):",
                list(METHOD_NAMES.keys()),
                format_func=lambda k: METHOD_NAMES[k],
                index=4,
                horizontal=False
            )

            all_method_chunks = load_chunks_for_method(current_doc["doc_id"], selected_method_key)
            page_chunks = [c for c in all_method_chunks if current_page in c.get("source_pages", [])]

            st.markdown(
                f"**Tổng số Chunks của phương pháp trong tài liệu:** `{len(all_method_chunks)}` | "
                f"**Số Chunks xuất hiện tại Trang {current_page}:** `{len(page_chunks)}`"
            )

            if not page_chunks:
                st.info(f"Không có Chunk nào của phương pháp này chứa trang {current_page}.")
            else:
                for idx, ch in enumerate(page_chunks, 1):
                    chunk_id = ch.get("chunk_id", f"chunk_{idx}")
                    token_count = ch.get("token_count", 0)
                    chunk_type = ch.get("chunk_type", "text")
                    is_frag = ch.get("has_table_fragmentation", False)
                    src_pages = ch.get("source_pages", [])

                    frag_badge = "<span class='badge badge-red'>⚠️ VỠ BẢNG</span>" if is_frag else "<span class='badge badge-green'>✅ BẢO TOÀN BẢNG</span>"
                    type_badge = f"<span class='badge badge-blue'>{chunk_type.upper()}</span>"
                    pages_badge = f"<span class='badge badge-purple'>Trang: {src_pages}</span>"

                    with st.expander(f"🧩 Chunk #{idx}: {chunk_id} ({token_count} tokens) - {chunk_type}", expanded=(idx == 1)):
                        st.markdown(f"{type_badge} {pages_badge} {frag_badge}", unsafe_allow_html=True)
                        st.markdown(f"**Token Count:** `{token_count}` | **Mã Chunk:** `[{chunk_id}]`")

                        c_tab1, c_tab2, c_tab3 = st.tabs([
                            "📜 Nội Dung Đầy Đủ (Content / Generation)",
                            "🎯 Nội Dung Truy Xuất (Retrieval / Fact Tuples)",
                            "⚙️ Metadata Chi Tiết"
                        ])

                        with c_tab1:
                            st.markdown(ch.get("content", ""))

                        with c_tab2:
                            retrieval_txt = ch.get("content_retrieval")
                            if retrieval_txt:
                                st.code(retrieval_txt, language="markdown")
                            else:
                                st.info("Phương pháp này không tạo trường biểu diễn kép `content_retrieval` riêng biệt.")

                        with c_tab3:
                            st.json(ch.get("metadata", {}))

        with main_tab3:
            st.markdown(f"#### ⚖️ Đối Sánh 5 Chiến Lược Phân Đoạn Trên Trang {current_page}")

            comparison_rows = []
            for m_key, m_name in METHOD_NAMES.items():
                m_chunks = load_chunks_for_method(current_doc["doc_id"], m_key)
                m_page_chunks = [c for c in m_chunks if current_page in c.get("source_pages", [])]
                frag_count = sum(1 for c in m_page_chunks if c.get("has_table_fragmentation", False))
                total_toks = sum(c.get("token_count", 0) for c in m_page_chunks)
                chunk_ids = [c.get("chunk_id", "") for c in m_page_chunks]

                comparison_rows.append({
                    "Phương Pháp": m_name,
                    "Số Chunks trên trang": len(m_page_chunks),
                    "Tổng Tokens": total_toks,
                    "Bị Vỡ Bảng?": "❌ Có vỡ bảng" if frag_count > 0 else "✅ Nguyên vẹn",
                    "Danh Sách Chunk IDs": ", ".join(chunk_ids[:3]) + (f" (+{len(chunk_ids)-3})" if len(chunk_ids) > 3 else "")
                })

            df_comp = pd.DataFrame(comparison_rows)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Bottom Section: Gold Benchmark Ground Truth Inspection
    st.markdown("---")
    st.markdown("### 🎯 Đối Soát Tập Benchmark Chuẩn (Gold Test Set Ground Truth)")

    current_gold_questions = gold_by_doc_page.get((current_doc["doc_id"], current_page), [])

    if current_gold_questions:
        st.success(f"🎯 **Phát hiện {len(current_gold_questions)} câu hỏi Benchmark Ground Truth được trích xuất từ Trang {current_page}!**")

        for idx, q in enumerate(current_gold_questions, 1):
            qid = q.get("id", f"GOLD_{idx}")
            diff = q.get("difficulty", "L1")
            qtype = q.get("question_type", "factual")
            gold_ans = q.get("gold_answer", "")
            norm_val = q.get("normalized_value_in_million")
            facts = q.get("financial_facts", [])
            evidence = q.get("gold_evidence", [])
            gold_chunks = q.get("gold_chunk_ids", {})
            coverage = q.get("coverage", {})
            containment = q.get("containment_scores", {})

            with st.expander(f"📌 [{qid}] {q.get('question')} ({diff} - {qtype})", expanded=True):
                col_q1, col_q2 = st.columns([1.2, 1], gap="medium")

                with col_q1:
                    st.markdown(f"**Câu hỏi:**")
                    st.markdown(f"### {q.get('question')}")

                    st.markdown("<div class='gold-box'>", unsafe_allow_html=True)
                    st.markdown(f"**🏆 Ground Truth Answer (Đã xác minh 100% khớp SEC 10-K):**")
                    st.markdown(f"<h3 style='color:#15803D; margin-top:0.3rem;'>{gold_ans}</h3>", unsafe_allow_html=True)
                    if norm_val is not None:
                        st.markdown(f"Giá trị chuẩn hóa: <code>{norm_val} million USD</code>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if facts:
                        st.markdown("**Thông số tài chính chuẩn (Financial Facts):**")
                        df_facts = pd.DataFrame(facts)
                        st.dataframe(df_facts, use_container_width=True, hide_index=True)

                    if evidence:
                        st.markdown("**Bằng chứng gốc (Evidence Text & Keywords):**")
                        for ev in evidence:
                            st.info(f"**Section:** {ev.get('section', 'N/A')} | **Keywords:** `{', '.join(ev.get('keywords', []))}`\n\n_{ev.get('text', '')}_")

                with col_q2:
                    st.markdown("**Đối Soát Độ Phủ Của 5 Phương Pháp Chunking (Alignment):**")

                    rows_align = []
                    for m_key, m_name in METHOD_NAMES.items():
                        assigned_chunks = gold_chunks.get(m_key, [])
                        cov_status = coverage.get(m_key, "N/A")
                        c_score = containment.get(m_key, 0.0)

                        cov_badge = "🟢 Full" if cov_status == "full" else ("🟡 Partial" if cov_status == "partial" else "🔴 Miss")

                        rows_align.append({
                            "Phương Pháp": m_name.split(":")[0],
                            "Gold Chunk ID": ", ".join(assigned_chunks) if assigned_chunks else "None",
                            "Coverage": cov_badge,
                            "Containment": f"{c_score:.2f}" if isinstance(c_score, (int, float)) else str(c_score)
                        })

                    df_align = pd.DataFrame(rows_align)
                    st.dataframe(df_align, use_container_width=True, hide_index=True)

    else:
        st.info(
            f"ℹ️ Trang {current_page} không có câu hỏi Benchmark Ground Truth trực tiếp. "
            f"Tài liệu **{current_doc['display_name']}** có tổng cộng **{len(doc_gold_list)} câu hỏi Benchmark**."
        )

# ==============================================================================
# MODE 2: RETRIEVAL STUDIO (5-METHOD PARALLEL COMPARISON ARENA)
# ==============================================================================
else:
    st.markdown("<div class='main-header'>🔍 FinAnalyst-AI: Retrieval Studio (Dense Search Benchmark)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Đấu trường thực nghiệm đối soát năng lực truy xuất thông tin tài chính "
        "trên <b>toàn bộ 5 phương pháp phân đoạn</b> (Dense Only Baseline - BAAI/bge-base-en-v1.5).</div>",
        unsafe_allow_html=True
    )

    studio_tab1, studio_tab2 = st.tabs([
        "⚔️ Đấu Trường Đối Sánh Song Song 5 Phương Pháp",
        "📊 Bảng Xếp Hạng & Biểu Đồ Benchmark (140 Queries)"
    ])

    # --------------------------------------------------------------------------
    # Sub-tab 1: Interactive Parallel Search Arena
    # --------------------------------------------------------------------------
    with studio_tab1:
        st.markdown("### 🎛️ Bộ Điều Khiển Truy Vấn (Query Console)")

        input_mode = st.radio(
            "Phương thức chọn câu hỏi:",
            [
                "⭐ Chọn câu hỏi chuẩn từ Tập Benchmark Ground Truth (140 câu)",
                "✍️ Nhập câu hỏi tự do (Ad-hoc Natural Language Query)"
            ],
            horizontal=True
        )

        selected_gold_q = None
        user_query = ""

        if "⭐" in input_mode:
            f_col1, f_col2 = st.columns([1, 1])
            with f_col1:
                comp_options = ["ALL"] + sorted(list(set(q.get("target_company") or q.get("document", {}).get("company", "") for q in all_gold_questions if q.get("target_company") or q.get("document", {}).get("company"))))
                selected_comp = st.selectbox("Lọc theo Doanh Nghiệp:", comp_options, index=0)

            with f_col2:
                diff_options = ["ALL", "L1 (Direct Fact)", "L2 (Table Reasoning)", "L3 (Multi-Year)", "L4 (Qualitative)", "L5 (Abstention)"]
                selected_diff = st.selectbox("Lọc theo Độ Khó:", diff_options, index=0)

            # Filter questions
            filtered_qs = []
            for q in all_gold_questions:
                c = q.get("target_company") or q.get("document", {}).get("company", "")
                d = q.get("difficulty", "L1")

                comp_ok = (selected_comp == "ALL" or c.upper() == selected_comp.upper())
                diff_prefix = selected_diff.split()[0]
                diff_ok = (selected_diff == "ALL" or d == diff_prefix)

                if comp_ok and diff_ok:
                    filtered_qs.append(q)

            if not filtered_qs:
                st.warning("Không tìm thấy câu hỏi phù hợp bộ lọc.")
            else:
                q_options = {
                    f"[{q.get('id')}] ({q.get('difficulty')} - {q.get('target_company') or q.get('document',{}).get('company')}): {q.get('question')}": q
                    for q in filtered_qs
                }
                chosen_label = st.selectbox("Chọn câu hỏi thử nghiệm:", list(q_options.keys()))
                selected_gold_q = q_options[chosen_label]
                user_query = selected_gold_q.get("question", "")

                # Display Gold Reference Box
                with st.expander("📌 Xem Thông Tin Chuẩn Vàng (Ground Truth Reference)", expanded=True):
                    g_c1, g_c2 = st.columns([1.2, 1])
                    with g_c1:
                        st.markdown(f"**Câu hỏi:** {selected_gold_q.get('question')}")
                        st.markdown("<div class='gold-box'>", unsafe_allow_html=True)
                        st.markdown(f"**🏆 Ground Truth Answer:**")
                        st.markdown(f"<h3 style='color:#15803D; margin-top:0.2rem;'>{selected_gold_q.get('gold_answer')}</h3>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        facts = selected_gold_q.get("financial_facts", [])
                        if facts:
                            st.markdown("**Financial Facts:**")
                            st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)

                    with g_c2:
                        doc_d = selected_gold_q.get("document", {})
                        st.markdown(f"**Hồ sơ:** `{doc_d.get('company')} - {doc_d.get('reporting_period')} 10-K`")
                        evs = selected_gold_q.get("gold_evidence", [])
                        if evs:
                            st.markdown(f"**Trang chứa bằng chứng:** `Trang {', '.join(str(e.get('page')) for e in evs)}`")
                            st.markdown(f"**Từ khóa định danh:** `{', '.join(evs[0].get('keywords', []))}`")

                        g_chunks = selected_gold_q.get("gold_chunk_ids", {})
                        if g_chunks:
                            st.markdown("**Mã Chunk chuẩn đối soát (Gold Chunk IDs):**")
                            align_data = []
                            for m in RETRIEVAL_METHODS:
                                m_ids = g_chunks.get(m["key"], [])
                                align_data.append({
                                    "Phương Pháp": m["display"],
                                    "Expected Chunk ID": ", ".join(m_ids) if m_ids else "N/A"
                                })
                            st.dataframe(pd.DataFrame(align_data), use_container_width=True, hide_index=True)

        else:
            user_query = st.text_input(
                "Nhập câu hỏi tài chính (tiếng Anh hoặc tiếng Việt):",
                value="What was Apple's total net sales for the fiscal year ended September 28, 2024?",
                placeholder="VD: What was Nvidia's data center revenue in fiscal year 2025?"
            )

        # Settings row
        s_c1, s_c2, s_c3, s_c4 = st.columns([1.3, 0.9, 1.0, 1.2])
        with s_c1:
            search_mode = st.radio(
                "Chế độ Tìm kiếm (Search Engine):",
                ["hybrid", "dense"],
                format_func=lambda x: "🔥 V1 Hybrid (Dense + BM25 RRF)" if x == "hybrid" else "⚡ V0 Pure Dense (BGE Baseline)",
                index=0,
            )
        with s_c2:
            top_k_select = st.slider("Số lượng Top Chunks (Top-K):", min_value=1, max_value=10, value=5)
        with s_c3:
            use_filter = st.checkbox("Áp dụng Filter Ticker", value=False)
            filter_ticker = None
            if use_filter:
                filter_ticker = st.selectbox("Chọn Ticker lọc:", ["AAPL", "AMZN", "AMD", "INTC", "NKE", "NVDA", "WMT"])
        with s_c4:
            st.markdown("<br>", unsafe_allow_html=True)
            run_search_btn = st.button("🚀 THỰC THI TRUY XUẤT ĐỐI SÁNH 5 PHƯƠNG PHÁP", type="primary", use_container_width=True)

        st.markdown("---")

        # Execute Search across all 5 methods
        if run_search_btn or user_query:
            retriever = get_dense_retriever()

            st.markdown(f"### 🎯 Kết Quả Đối Sánh Song Song 5 Phương Pháp (Top {top_k_select} Chunks | Chế độ: **{search_mode.upper()}**)")
            st.markdown(f"**Truy vấn:** *\"{user_query}\"*")

            # 5 Columns for 5 Methods
            method_cols = st.columns(5, gap="small")

            for col_idx, m_spec in enumerate(RETRIEVAL_METHODS):
                m_key = m_spec["key"]
                m_display = m_spec["display"]
                m_badge_class = m_spec["badge_class"]
                m_badge_text = m_spec["badge_text"]
                coll_name = get_collection_name(DEFAULT_EMBEDDING_PROVIDER, m_key)

                with method_cols[col_idx]:
                    st.markdown(f"#### {m_display}")
                    st.markdown(f"<span class='badge {m_badge_class}'>{m_badge_text}</span>", unsafe_allow_html=True)

                    # Search execution
                    start_t = time.perf_counter()
                    try:
                        results = retriever.search(
                            collection_name=coll_name,
                            query=user_query,
                            top_k=top_k_select,
                            mode=search_mode,
                            ticker=filter_ticker if use_filter else None
                        )
                        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                        gc.collect()
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                        results = []
                        elapsed_ms = 0.0

                    # Check ground truth hit
                    hit_rank = None
                    if selected_gold_q:
                        for r in results:
                            if is_chunk_relevant(r.payload, selected_gold_q, m_key):
                                hit_rank = r.rank
                                break

                    if hit_rank is not None:
                        st.markdown(
                            f"<div class='retrieval-hit-box'>"
                            f"<b>🏆 GROUND TRUTH HIT!</b><br>"
                            f"Xuất hiện tại <b>Rank #{hit_rank}</b><br>"
                            f"<small>⏱️ Độ trễ: {elapsed_ms:.1f} ms</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='retrieval-miss-box'>"
                            f"<b>❌ KHÔNG HIT TRONG TOP-{top_k_select}</b><br>"
                            f"<small>⏱️ Độ trễ: {elapsed_ms:.1f} ms</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    # Render top-k chunk cards
                    for r in results:
                        is_rel = (selected_gold_q and is_chunk_relevant(r.payload, selected_gold_q, m_key))
                        card_class = "chunk-card hit-card" if is_rel else "chunk-card"
                        hit_tag = "<span class='badge badge-green'>🏆 GROUND TRUTH CHUNK</span><br>" if is_rel else ""

                        st.markdown(f"""
                        <div class='{card_class}'>
                            {hit_tag}
                            <b>Rank #{r.rank}</b> | Score: <b>{r.score:.4f}</b><br>
                            <code>[{r.chunk_id}]</code><br>
                            <small>Trang: {r.payload.get('source_pages')} | Type: {r.payload.get('chunk_type', 'N/A')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander(f"📖 Chi tiết Rank #{r.rank} ({r.chunk_id})"):
                            r_tab1, r_tab2 = st.tabs(["📜 Content (Bảng/Văn bản)", "🎯 Fact Tuples / Retrieval"])
                            with r_tab1:
                                st.markdown(r.payload.get("content", ""))
                            with r_tab2:
                                r_text = r.payload.get("content_retrieval")
                                if r_text:
                                    st.code(r_text, language="markdown")
                                else:
                                    st.info("Phương pháp này không có trường `content_retrieval` riêng.")

            # Summary Verdict below 5 columns
            st.markdown("---")
            st.markdown("### 💡 Phân Tích Thực Nghiệm Chuyên Sâu:")
            v_col1, v_col2 = st.columns([1, 1])
            with v_col1:
                st.markdown("""
                - **Method 5 (Proposed Golden Hybrid):**
                  + Bảo toàn cấu trúc bảng Markdown trọn vẹn, không cắt ngang giữa các dòng/cột.
                  + Sử dụng biểu diễn kép **Fact Tuples** (`content_retrieval`) giúp mô hình vector Bi-Encoder nhận diện chính xác cặp quan hệ `(Thực thể, Chỉ số, Niên độ, Giá trị)`, đạt tỷ lệ trúng bảng cao nhất.
                """)
            with v_col2:
                st.markdown("""
                - **Method 1 & Method 2 (Fixed-Size / Deterministic):**
                  + Thường xuyên bị phân mảnh (vỡ bảng) do ranh giới 512 token cắt ngang bảng tài chính lớn.
                  + Mất tiêu đề cột hoặc hàng thuyết minh dẫn đến hiện tượng **mồ côi dữ liệu (orphaned rows)**, khiến mô hình vector bị trôi dạt ngữ nghĩa và trượt khỏi Top-K.
                """)

    # --------------------------------------------------------------------------
    # Sub-tab 2: Benchmark Leaderboard & Interactive Visual Charts
    # --------------------------------------------------------------------------
    with studio_tab2:
        st.markdown("### 🏆 Bảng Xếp Hạng & Biểu Đồ Benchmark Định Lượng (140 Queries)")
        st.markdown(
            "Dữ liệu được trích xuất trực tiếp từ kết quả chạy thực nghiệm tự động trên GPU "
            "tại file [`outputs/retrieval/v0_dense_baseline/dense_baseline_report.md`](file:///c:/Users/ThanhDz/Downloads/DATN_Finance/FinAnalyst-AI/outputs/retrieval/v0_dense_baseline/dense_baseline_report.md)."
        )

        if not benchmark_summary:
            st.warning("Chưa tìm thấy file tổng kết `v0_retrieval_evaluation_summary.json`. Vui lòng chạy Bước 1.2 trước.")
        else:
            # 5 KPI Cards for the 5 Methods
            kpi_cols = st.columns(5)
            for idx, m_spec in enumerate(RETRIEVAL_METHODS):
                m_key = m_spec["key"]
                m_data = benchmark_summary.get(m_key, {})
                ov = m_data.get("overall", {})

                with kpi_cols[idx]:
                    st.metric(
                        label=m_spec["display"],
                        value=f"Hit@10: {ov.get('hit@10', 0.0) * 100:.1f}%",
                        delta=f"MRR: {ov.get('mrr', 0.0):.4f}"
                    )
                    st.caption(f"Hit@5: **{ov.get('hit@5', 0.0)*100:.1f}%** | Latency: **{ov.get('avg_latency_ms', 0.0):.1f}ms**")

            st.markdown("---")

            # Comparative Charts
            chart_col1, chart_col2 = st.columns([1.2, 1])

            with chart_col1:
                st.markdown("#### 📊 So Sánh Các Chỉ Số Truy Xuất Cốt Lõi (Hit@K & MRR)")
                chart_rows = []
                for m_spec in RETRIEVAL_METHODS:
                    ov = benchmark_summary.get(m_spec["key"], {}).get("overall", {})
                    chart_rows.append({
                        "Phương Pháp": m_spec["display"].replace("Method ", "M"),
                        "Hit@1 (%)": ov.get("hit@1", 0.0) * 100,
                        "Hit@5 (%)": ov.get("hit@5", 0.0) * 100,
                        "Hit@10 (%)": ov.get("hit@10", 0.0) * 100,
                        "Hit@20 (%)": ov.get("hit@20", 0.0) * 100,
                        "MRR x100": ov.get("mrr", 0.0) * 100,
                    })

                df_metrics = pd.DataFrame(chart_rows).set_index("Phương Pháp")
                st.bar_chart(df_metrics[["Hit@5 (%)", "Hit@10 (%)", "Hit@20 (%)", "MRR x100"]], height=350)

            with chart_col2:
                st.markdown("#### 🎯 Đột Phá Của Method 5 Theo Độ Khó (Hit@10 / Recall@10)")
                diff_rows = []
                for m_spec in RETRIEVAL_METHODS:
                    bd = benchmark_summary.get(m_spec["key"], {}).get("by_difficulty", {})
                    diff_rows.append({
                        "Phương Pháp": m_spec["display"].replace("Method ", "M"),
                        "L1 (Direct Fact)": bd.get("L1", {}).get("recall@10", 0.0) * 100,
                        "L2 (Table Reasoning)": bd.get("L2", {}).get("recall@10", 0.0) * 100,
                        "L3 (Multi-Year)": bd.get("L3", {}).get("recall@10", 0.0) * 100,
                        "L4 (Qualitative)": bd.get("L4", {}).get("recall@10", 0.0) * 100,
                    })

                df_diff = pd.DataFrame(diff_rows).set_index("Phương Pháp")
                st.bar_chart(df_diff, height=350)

            # Master Evaluation Table
            st.markdown("#### 📋 Bảng Dữ Liệu Chi Tiết Đầy Đủ")
            table_rows = []
            for m_spec in RETRIEVAL_METHODS:
                ov = benchmark_summary.get(m_spec["key"], {}).get("overall", {})
                table_rows.append({
                    "Phương Pháp Chunking": m_spec["display"],
                    "Đặc Điểm Phân Đoạn": m_spec["desc"],
                    "Hit@1": f"{ov.get('hit@1', 0.0) * 100:.1f}%",
                    "Hit@3": f"{ov.get('hit@3', 0.0) * 100:.1f}%",
                    "Hit@5": f"{ov.get('hit@5', 0.0) * 100:.1f}%",
                    "Hit@10": f"{ov.get('hit@10', 0.0) * 100:.1f}%",
                    "Hit@20": f"{ov.get('hit@20', 0.0) * 100:.1f}%",
                    "MRR": f"{ov.get('mrr', 0.0):.4f}",
                    "NDCG@10": f"{ov.get('ndcg@10', 0.0):.4f}",
                    "Độ Trễ Trung Bình": f"{ov.get('avg_latency_ms', 0.0):.1f} ms"
                })

            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
