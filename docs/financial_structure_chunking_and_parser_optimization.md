# Kiến Trúc Structure Chunking & Tối Ưu Hóa Parser BCTC Cho Financial RAG

> **Tài liệu đặc tả kỹ thuật (Technical Specification & Architecture Design) — Bản V2**  
> **Dự án:** Trợ lý Nghiên cứu & Phân tích Báo cáo Tài chính Việt Nam (DATN Finance)  
> **Trọng tâm:** Khắc phục nhược điểm của Vision LLM (khoảng trắng `&nbsp;`, vỡ ngữ cảnh trang) và thiết kế hệ thống **Structure Chunking (Semantic Financial Chunking)** chuyên sâu cho Báo cáo tài chính (BCTC/BCTN).

> [!NOTE]
> Bản V2 đã sửa toàn bộ 8 vấn đề được phát hiện trong bản V1:
> - Bug regex sanitizer xóa nhầm dữ liệu thật (nghiêm trọng nhất).
> - Thứ tự rule regex phá hủy thụt lề phân cấp.
> - Số liệu ví dụ không nhất quán.
> - Stitcher thiếu fallback cho lỗi OCR.
> - Table-Atomic Chunk thiếu xử lý bảng cực lớn.
> - Metadata schema chưa nối với hệ thống QC/Evaluation.
> - Thiếu chunk cấp tài liệu cho truy vấn liên-note.
> - Action Plan thiếu A/B test rủi ro compliance của V5.

---

## Mục lục

1. [Bối cảnh & Vấn đề cốt tử](#1-bối-cảnh--vấn-đề-cốt-tử)
2. [Chuẩn hóa định dạng đầu ra Parser](#2-chuẩn-hóa-định-dạng-đầu-ra-parser)
3. [Cấu trúc lại Prompt Parser (V5 Directive)](#3-cấu-trúc-lại-prompt-parser-v5-directive)
4. [Kiến trúc Structure Chunking](#4-kiến-trúc-structure-chunking)
5. [Biểu diễn kép cho Chunk (Dual Representation)](#5-biểu-diễn-kép-cho-chunk)
6. [Schema Metadata chuẩn hóa cho Vector DB](#6-schema-metadata-chuẩn-hóa-cho-vector-db)
7. [Lộ trình triển khai (Action Plan)](#7-lộ-trình-triển-khai)

---

## 1. Bối cảnh & Vấn đề Cốt tử (Problem Statement)

Trong quá trình thử nghiệm thực tế bóc tách dữ liệu từ file BCTC gốc (điển hình như trang 50–54 của file `VHM_BCTC_2025_HopNhat.pdf`), việc phân tích chi tiết kết quả trả về từ mô hình thị giác (Vision LLM — DeepSeek Vision / Gemini) đã chỉ ra 2 rào cản kỹ thuật nghiêm trọng đối với hệ thống RAG:

### 1.1 Hiện tượng "Bẫy bố cục thị giác" và Rác Token (`&nbsp;`)
- **Nguyên nhân:** Vision LLM nhận ảnh đầu vào và cố gắng tái hiện bố cục không gian thị giác (visual 2D layout) trên môi trường văn bản 1D. Khi gặp bảng biểu hoặc tiêu đề có các phần tử nằm dạt sang hai bên (ví dụ: Tên công ty bên trái, mã biểu mẫu `B09-DN/HN` bên phải), mô hình chèn hàng chục ký tự `&nbsp;` hoặc dấu cách liên tiếp.
- **Tác hại đến RAG:**
  1. **Lãng phí Token Budget:** Mỗi ký tự `&nbsp;` chiếm 6 ký tự thô. Chuỗi 30–50 `&nbsp;` ngốn tới **25–40 tokens** hoàn toàn vô nghĩa trong ngữ cảnh LLM.
  2. **Làm nhiễu Vector nhúng (Embedding Dilution):** Các mô hình Dense Embedding (BGE-M3, OpenAI `text-embedding-3`, Cohere) tính trọng số ngữ nghĩa trên toàn chuỗi. Sự xuất hiện dày đặc của token rác làm méo mó vector không gian, làm giảm đáng kể điểm tương đồng Cosine Similarity khi truy vấn thông tin nghiệp vụ.
  3. **Vỡ ranh giới từ vựng trong BM25 (Sparse Search):** BM25 coi chuỗi ký tự rác này là các term đặc biệt, gây nhiễu chỉ số IDF (Inverse Document Frequency).

### 1.2 Ranh giới trang in (Page Break) phá hủy tính toàn vẹn ngữ nghĩa

BCTC được trình bày theo cấu trúc logic: **Chương/Mục → Thuyết minh → Bảng số liệu tổng hợp → Chú thích chi tiết (Footnotes)**. Ranh giới trang PDF chỉ là giới hạn vật lý của khổ giấy in (A4 dọc hoặc A4 ngang), **hoàn toàn không tương thích với ranh giới ngữ nghĩa**.

**Ví dụ thực tế VHM 2025 — Thuyết minh 13 (Tài sản khác):**
- **Trang 51 (In: 43):** Chứa bảng số liệu tổng hợp TM13 gồm 2 phần:
  - Ngắn hạn — TỔNG CỘNG: `88.872.716 triệu VND` (Số cuối năm), `25.843.774 triệu VND` (Số đầu năm)
  - Dài hạn — TỔNG CỘNG: `67.341.923 triệu VND` (Số cuối năm), `40.471.695 triệu VND` (Số đầu năm)
  - Trong đó phần Dài hạn gồm: *Đặt cọc đầu tư (ii): `66.309.587`* + *Đặt cọc thương mại (iii): `1.032.336`* + *Tài sản khác: `-`* = `67.341.923`
  - Kèm mã tham chiếu `(i), (ii), (iii)` nhưng **chưa có lời giải thích**.
- **Trang 52 (In: 44):** Chứa toàn bộ nội dung thuyết minh chi tiết cho `(i), (ii), (iii)`: nêu rõ khoản đặt cọc 12.000 tỷ là giải phóng mặt bằng dự án nào, khoản 53.402 tỷ là cho bên liên quan nào.

**Hậu quả nếu cắt chunk theo trang hoặc theo kích thước cố định (Fixed-size 512 tokens):**
- Retriever tìm thấy bảng số liệu ở trang 51 nhưng **thiếu hoàn toàn lời giải thích ở trang 52**.
- LLM trả lời câu hỏi: *"Khoản đặt cọc lớn nhất của Vinhomes năm 2025 dành cho mục đích gì?"* → **Thất bại hoặc bịa đặt (Hallucination)** vì không có ngữ cảnh nối tiếp.

---

## 2. Chuẩn hóa Định dạng Đầu ra Parser (Sanitization & Representation)

### 2.1 Sử dụng HTML Tags có cấu trúc kết hợp Markdown

Các mô hình ngôn ngữ lớn hiện đại (DeepSeek, GPT-4o, Claude 3.5, Gemini 2.0) đều được pre-train trên hàng tỷ trang web HTML. HTML cung cấp cú pháp phân cấp hình cây (DOM Tree) mà Markdown thuần không thể thể hiện được (như phân cấp thẻ, thuộc tính metadata, liên kết cha con). **Rất khuyến nghị cho BCTC.**

**Đề xuất định dạng lai (Hybrid Semantic Markdown + HTML Tags):**
```html
<section class="bctc-note" id="note-13" name="TÀI SẢN KHÁC" page-start="51" page-end="52">
  <div class="note-header">
    <h3>13. TÀI SẢN KHÁC</h3>
    <span class="unit" currency="VND" scale="triệu">Đơn vị tính: triệu VND</span>
  </div>

  <table class="financial-table" id="note-13-table-summary">
    <thead>
      <tr>
        <th>Khoản mục</th>
        <th>Số cuối năm (31/12/2025)</th>
        <th>Số đầu năm (01/01/2025)</th>
      </tr>
    </thead>
    <tbody>
      <tr class="group-header"><td colspan="3"><b>Ngắn hạn</b></td></tr>
      <tr>
        <td>Đặt cọc cho mục đích đầu tư <ref target="#fn-13-i">(i)</ref></td>
        <td class="num">88.791.672</td>
        <td class="num">25.626.349</td>
      </tr>
      ...
    </tbody>
  </table>

  <div class="note-footnotes">
    <div class="footnote-item" id="fn-13-i">
      <b>(i)</b> Các khoản đặt cọc dự án tương lai: 12.000 tỷ VND giải phóng mặt bằng...
    </div>
  </div>
</section>
```

### 2.2 Quy trình Làm sạch Tự động (Sanitization Pipeline)

> [!CAUTION]
> **Sửa lỗi nghiêm trọng từ bản V1:** Bản V1 dùng regex đoán mò trên nội dung đã làm phẳng để lọc header/footer trang:
> ```python
> # ⚠️ BUG NGHIÊM TRỌNG — ĐÃ XÓA TRONG V2 ⚠️
> text = re.sub(r'^\s*(?:Công ty Cổ phần|B09-DN/HN|Mẫu số).*$', '', text, flags=re.MULTILINE)
> ```
> Regex này sẽ **xóa nhầm dữ liệu thật** vì "Công ty Cổ phần" là tiền tố tên doanh nghiệp cực kỳ phổ biến, xuất hiện dưới dạng nội dung nghiệp vụ hợp lệ. Ví dụ thật từ MWG:
> - *"Công ty Cổ phần Thế Giới Số Trần Anh"* (TM16 — Lợi thế thương mại)
> - *"Công ty Cổ phần Thương mại Bách Hóa Xanh"* (Danh sách công ty con)
> - Hàng loạt tên đối tác trong TM6/TM7/TM9/TM18/TM31
>
> **Gốc rễ vấn đề:** Lọc header/footer bằng regex nội dung trên text đã làm phẳng là sai phương pháp. Cần giải quyết **từ gốc, ngay tại tầng Parser** thay vì vá víu ở tầng post-processing.

**Giải pháp đúng: Tách header/footer bằng Block-type Tagging từ Parser**

Thay vì để mọi dòng text ra cùng một luồng rồi cố đoán dòng nào là header lặp, ta yêu cầu Parser (prompt V5) **gắn thẻ phân loại ngay từ lúc parse**, tương tự cách MinerU gán `page_header / page_footer / page_number` thành block-type riêng:

```text
(Trong SYSTEM_PROMPT V5)
═══ HEADER/FOOTER TRANG LẶP LẠI ═══
• Nếu trang có dòng lặp lại ở mọi trang (Tên công ty, Mã biểu mẫu B09-DN/HN,
  dòng "THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT (tiếp theo)", ngày chốt số liệu),
  bọc chúng trong thẻ comment:
  <!-- page_header: Công ty Cổ phần Vinhomes | B09-DN/HN -->
  <!-- page_subheader: THUYẾT MINH BCTC HỢP NHẤT (tiếp theo) | 31/12/2025 -->
• Chỉ giữ lại nội dung mới (tiêu đề thuyết minh, bảng, đoạn văn) dưới dạng
  Markdown chuẩn.
```

Và trong canonical JSON, mỗi block có định danh duy nhất (`block_id`) và trường `block_type` rõ ràng:
```json
{
  "block_id": "p51_b01",
  "block_type": "page_header",
  "text": "Công ty Cổ phần Vinhomes | B09-DN/HN"
}
```

Đối với bảng biểu, gán thêm `table_id` duy nhất theo trang:
```json
{
  "block_id": "p51_b05",
  "block_type": "table",
  "table_id": "p51_t02",
  "headers": ["Khoản mục", "Số cuối năm", "Số đầu năm"],
  "rows": [...]
}
```
> [!TIP]
> `table_id` giúp module Stitcher ở tầng sau dễ dàng ánh xạ ghép các bảng vắt qua nhiều trang:
> `p51_t02` + `p52_t01` → cùng một bảng logic:
> ```json
> {
>   "logical_table_id": "note13_table01",
>   "source_tables": ["p51_t02", "p52_t01"]
> }
> ```

**Tách bạch giữa Metadata và Content:**
Thay vì chỉ ghi text thô vào Markdown rồi cố trích xuất ngược, Canonical JSON lưu cấu trúc metadata chuẩn hóa:
```json
{
  "unit": {
    "raw": "triệu VND",
    "value": "VND",
    "scale": "million"
  },
  "period": {
    "raw": "31/12/2025 vs 01/01/2025",
    "current": "31/12/2025",
    "previous": "01/01/2025"
  },
  "section_codes": ["12", "13"],
  "section_context_candidates": ["13"]
}
```
Từ Canonical JSON này, pipeline sinh Markdown (`content_generation` / `content_retrieval`) sẽ render thành:
`[Đơn vị tính: triệu VND] | [Kỳ báo cáo: 31/12/2025 vs 01/01/2025]`
Đảm bảo luồng dữ liệu một chiều: **Canonical JSON (Semantics & Metadata) → Markdown Representation**, không để cú pháp format Markdown chi phối ngược lại ngữ nghĩa dữ liệu.

Khi post-processing (sanitization), ta lọc **theo `block_type`** thay vì đoán regex:
```python
# AN TOÀN — lọc theo loại block, không lọc theo nội dung khớp mẫu
blocks = [b for b in page_blocks if b["block_type"] not in ("page_header", "page_footer", "page_number")]
```

**Pipeline sanitization cho nội dung Markdown thô (các rule còn lại):**

> [!IMPORTANT]
> **Sửa lỗi thứ tự từ bản V1:** Rule nhận diện thụt lề phân cấp **phải chạy TRƯỚC** rule san phẳng `&nbsp;`. Nếu đảo ngược, các mục con lồng sâu 3–4 cấp dùng ≥3 `&nbsp;` để thụt đầu dòng hợp lệ (phổ biến ở phần chính sách kế toán nhiều cấp) sẽ bị san phẳng thành 1 space trước khi kịp nhận diện cấp độ — mất luôn thông tin phân cấp cha-con.

```python
import re

def sanitize_financial_markdown(text: str) -> str:
    """
    Pipeline làm sạch Markdown trích xuất từ Vision LLM.
    THỨ TỰ CÁC RULE CÓ Ý NGHĨA — KHÔNG ĐƯỢC ĐẢO NGƯỢC.
    """

    # ── RULE 1 (TRƯỚC TIÊN): Nhận diện và chuẩn hóa thụt lề phân cấp ──
    # Phải chạy trước rule san phẳng &nbsp; để bảo toàn cấp độ lồng nhau.
    # Cấp 1: 1-2 &nbsp; + bullet -> "- "
    # Cấp 2: 3-6 &nbsp; + bullet -> "  - "
    # Cấp 3: 7-12 &nbsp; + bullet -> "    - "
    # Cấp 4: 13+ &nbsp; + bullet -> "      - "
    text = re.sub(r'^(&nbsp;){13,}\s*([▸\-\*•])\s*', '      - ', text, flags=re.MULTILINE)
    text = re.sub(r'^(&nbsp;){7,12}\s*([▸\-\*•])\s*', '    - ', text, flags=re.MULTILINE)
    text = re.sub(r'^(&nbsp;){3,6}\s*([▸\-\*•])\s*', '  - ', text, flags=re.MULTILINE)
    text = re.sub(r'^(&nbsp;){1,2}\s*([▸\-\*•])\s*', '- ', text, flags=re.MULTILINE)

    # ── RULE 2: San phẳng &nbsp; padding GIỮA DÒNG (không phải đầu dòng) ──
    # Chỉ áp dụng cho chuỗi &nbsp; liên tiếp NẰM SAU ký tự khác (mid-line padding).
    # Giữ nguyên &nbsp; đầu dòng nếu chưa được rule 1 xử lý.
    text = re.sub(r'(?<=\S)(&nbsp;|\s){3,}(?=\S)', ' ', text)

    # ── RULE 3: Chuẩn hóa dấu gạch ngang bullet còn sót ──
    text = re.sub(r'^\s*([▸])\s*', '- ', text, flags=re.MULTILINE)

    # ── RULE 4: Xóa các &nbsp; đầu dòng còn sót (không phải indent hợp lệ) ──
    text = re.sub(r'^(&nbsp;\s*)+(?![▸\-\*•])', '', text, flags=re.MULTILINE)

    # ── RULE 5: Bảo toàn nghiêm ngặt định dạng số tài chính ──
    # KHÔNG xóa dấu ngoặc đơn số âm: (123.456)
    # KHÔNG xóa dấu chấm ngăn cách nghìn: 123.456.789

    # ── RULE 6: Xóa các dòng trống thừa thãi (nhiều hơn 2 dòng trống) ──
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
```

> [!NOTE]
> **Header/Footer trang lặp:** KHÔNG xử lý bằng regex nội dung ở bước này. Đã được tách riêng từ Parser qua `block_type` tagging (xem trên). Hàm `sanitize_financial_markdown` chỉ xử lý vấn đề **định dạng** (`&nbsp;`, thụt lề, dòng trống), KHÔNG xử lý vấn đề **phân loại nội dung**.

---

## 3. Cấu trúc lại Prompt Parser (V5 Directive)

> [!WARNING]
> **Rủi ro quan trọng:** Prompt V5 thêm nhiều ràng buộc cấu trúc (heading hierarchy, `block_type` tagging, `<!-- CONTINUED_FROM_PREVIOUS_PAGE -->`) so với V4. Điều này tạo gánh nặng tuân thủ định dạng cao hơn lên model, đặc biệt ở đúng loại trang rủi ro cao nhất (trang xoay ngang, trang nhiều note liền nhau). Từ thực nghiệm V4 ta đã thấy Vision LLM tuân thủ instruction định dạng **không tuyệt đối 100%**.
>
> → **Bắt buộc chạy A/B test V4 vs V5** trên chính các trang khó trước khi áp dụng đại trà (xem [Mục 7.4](#74-bước-4--ab-test-v4-vs-v5-bắt-buộc-trước-khi-triển-khai)).

Cập nhật `SYSTEM_PROMPT` trong `parse_bctc_deepseek_v4.py` sang phiên bản **V5 Semantic & Structure-First**:

```text
═══ QUY TẮC PHÂN CẤP NGỮ NGHĨA (SEMANTIC HIERARCHY) ═══
• TUYỆT ĐỐI KHÔNG sử dụng &nbsp; hoặc khoảng trắng liên tiếp để căn lề thị giác.
• Sử dụng Markdown Heading chuẩn để phân định ranh giới:
  # Tên Báo cáo (BẢNG CÂN ĐỐI KẾ TOÁN, BÁO CÁO LƯU CHUYỂN TIỀN TỆ, THUYẾT MINH BCTC)
  ## Tên Thuyết minh cấp 1 (ví dụ: ## 11. HÀNG TỒN KHO, ## 13. TÀI SẢN KHÁC)
  ### Tên Thuyết minh con hoặc Bảng (ví dụ: ### 13.1 Đặt cọc ngắn hạn)
• Đơn vị tính: Luôn đặt ngay trước bảng biểu dưới dạng cú pháp:
  [Đơn vị tính: triệu VND] | [Kỳ báo cáo: 31/12/2025 vs 01/01/2025]

═══ HEADER/FOOTER TRANG LẶP LẠI ═══
• Nếu trang có dòng lặp lại ở mọi trang (Tên công ty, Mã biểu mẫu B09-DN/HN,
  dòng "THUYẾT MINH BÁO CÁO TÀI CHÍNH HỢP NHẤT (tiếp theo)", ngày chốt số liệu),
  bọc chúng trong thẻ comment:
  <!-- page_header: Công ty Cổ phần Vinhomes | B09-DN/HN -->
  <!-- page_subheader: THUYẾT MINH BCTC HỢP NHẤT (tiếp theo) | 31/12/2025 -->
• Chỉ giữ lại nội dung mới dưới dạng Markdown chuẩn. KHÔNG trộn lẫn nội dung
  dữ liệu thật và header lặp trang.

═══ XỬ LÝ TRANG TIẾP DIỄN (CONTINUED SECTIONS) ═══
• Nếu một Thuyết minh hoặc Bảng biểu bị cắt ngang từ trang trước và tiếp tục
  ở trang này:
  - Thêm thẻ metadata: <!-- CONTINUED_FROM_PREVIOUS_PAGE: section="13" -->
  - Tiếp tục nội dung phân cấp liền mạch, không ngắt quãng cấu trúc.

═══ BẢNG BIỂU & CHÚ THÍCH (FOOTNOTES) ═══
• Khi một ô dữ liệu có ký hiệu chú thích như (i), (ii), (*):
  - Giữ nguyên ký hiệu liền kề tên khoản mục: "Đặt cọc cho mục đích đầu tư (i)"
• Các đoạn thuyết minh văn bản giải thích cho (i), (ii) ở dưới bảng phải giữ
  nguyên mã đánh số:
  (i) Nội dung giải thích...
  (ii) Nội dung giải thích...
```

---

## 4. Kiến trúc Structure Chunking (Semantic Financial Chunker)

Thay vì dùng bộ cắt token ngây thơ (Naive Chunking), hệ thống xây dựng một **Semantic Financial Chunking Engine** bao gồm 4 tầng:

```text
       Raw PDF Pages
             │
             ▼
   [Vision Parser V5]  ── (Render 260 DPI + Auto-Rotate 90° Landscape)
             │
             ▼
    Parsed Raw Blocks   ── (Heading, Table, Paragraph, Footnote, Meta, page_header)
             │
             ▼
   ┌───────────────────────────────────────────────────────────┐
   │           DOCUMENT SECTION ASSEMBLER (STITCHER)           │
   │  • Ghép các trang cùng mã số thuyết minh (fuzzy match)   │
   │  • Nối các bảng đa trang (Multi-page Table Continuation)  │
   │  • Ghép danh sách Footnotes (i, ii) vào Bảng dữ liệu gốc │
   │  • Fallback: trang bắt đầu bằng (i)/(ii) + trang trước   │
   │    kết thúc dở dang → coi là tiếp diễn ngầm định         │
   └───────────────────────────────────────────────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────────────────────┐
   │             HIERARCHICAL CHUNKING STRATEGY                │
   ├───────────────────────────────────────────────────────────┤
   │ [Loại 1] Báo cáo chính (BS/IS/CF)                        │
   │          → Cắt theo nhóm (Tài sản / Nợ / Vốn)           │
   ├───────────────────────────────────────────────────────────┤
   │ [Loại 2] Thuyết minh (Note-level Chunk)                   │
   │          → 1 Thuyết minh = 1 Chunk + Footnotes đi kèm    │
   ├───────────────────────────────────────────────────────────┤
   │ [Loại 3] Bảng lớn vượt token limit                        │
   │          → Cắt theo nhóm phụ có sẵn trong bảng           │
   │            (VD: Ngắn hạn / Dài hạn), giữ header + unit   │
   ├───────────────────────────────────────────────────────────┤
   │ [Loại 4] Document Summary & Cross-Reference Index         │
   │          → 1 chunk tổng hợp cấp toàn báo cáo             │
   └───────────────────────────────────────────────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────────────────────┐
   │          CONTEXT INJECTION & DUAL REPRESENTATION          │
   │  • Tầng 1: Flattened Semantic Tuples (cho Dense/BM25)     │
   │  • Tầng 2: Clean Markdown/HTML Table (cho LLM Generation) │
   └───────────────────────────────────────────────────────────┘
```

### 4.1 Thiết kế Stitcher (Document Section Assembler) chịu lỗi OCR

> [!IMPORTANT]
> **Sửa lỗi từ bản V1:** Bản V1 giả định ghép nối bằng so khớp tiêu đề tuyệt đối. Trong thực nghiệm, ta đã quan sát lỗi OCR tên riêng (ví dụ: Lượm→Lượng, Trần→Trân). Cùng loại rủi ro hoàn toàn có thể xảy ra với số/tên thuyết minh — ví dụ "13" bị đọc nhầm thành "18" ở trang xoay ngang. Nếu Stitcher match bằng so khớp chuỗi tuyệt đối, một sai lệch nhỏ sẽ khiến 2 trang không được ghép mà không có cảnh báo.

**Cơ chế ghép nối 3 tầng (với fallback tăng dần):**

```text
Tầng 1 (Ưu tiên cao nhất) — Khớp theo MÃ SỐ THUYẾT MINH:
  • Trích xuất regex đa tầng:
    - Markdown heading: r"^#{1,3}\s*(\d+)[\.\s]"
    - Fallback bold heading: r"^\*{1,2}\s*(\d+)[\.\s]"
  • Khớp chuỗi số (ổn định hơn text tiêu đề rất nhiều).
  • Ví dụ: Trang 52 bắt đầu "**13. TÀI SẢN KHÁC** (tiếp theo)" → số "13"
    khớp với trang 51 cũng có heading số "13".

Tầng 2 (Phụ trợ) — Tín hiệu vị trí trang liền kề:
  • Nếu trang N và trang N+1 có cùng mã số thuyết minh → ghép.
  • Nếu mã số thuyết minh khác nhau nhưng trang N+1 có nhãn "(tiếp theo)"
    → ghép vào trang N (model có thể bỏ lỡ mã số nhưng giữ nhãn "tiếp theo").

Tầng 3 (Fallback ngầm định) — Heuristic tiếp diễn:
  • Nếu trang N+1 bắt đầu bằng "(i)", "(ii)", "(iii)" hoặc bullet list
    VÀ trang N kết thúc bằng bảng số liệu có ký hiệu (i), (ii)
    VÀ trang N+1 KHÔNG có heading thuyết minh mới (## <số>. <TÊN>)
    → Coi là tiếp diễn ngầm định, ghép vào cùng Node thuyết minh.
  • KHÔNG phụ thuộc 100% vào việc model có nhớ chèn đúng comment
    <!-- CONTINUED_FROM_PREVIOUS_PAGE --> hay không.

Cảnh báo:
  • Mọi trường hợp ghép ở Tầng 3 → đánh dấu flag "stitched_by_heuristic": true
    trong metadata chunk để review thủ công khi cần.
```

### 4.2 Quy tắc nhóm Chunk trong Báo cáo Tài chính

| Loại thành phần | Chiến lược | Ranh giới cắt | Ngữ cảnh tiêm (Injected Context) |
|---|---|---|---|
| **Báo cáo tài chính chính** *(Bảng cân đối, KQKD, LCTT)* | Statement Section Level | Tách theo nhóm chỉ tiêu lớn (A. TSNH; B. TSDH; C. Nợ phải trả; D. VCSH) | Tên công ty, Ngày chốt (31/12/2025), Đơn vị tính, Loại BC (Hợp nhất). |
| **Thuyết minh chuẩn** *(VD: TM11 Hàng tồn kho)* | Complete Note Level | Toàn bộ (Bảng chính + Bảng biến động + Thuyết minh chữ) | Mã TM, Tên TM, Đơn vị tính, Kỳ so sánh. |
| **Thuyết minh vắt trang** *(VD: TM13 bảng ở p.51, footnote ở p.52)* | Stitched Note Level | Ghép toàn bộ nội dung qua Stitcher thành 1 Node trước khi tính kích thước | Đầy đủ số tổng + chi tiết từng dự án từ cả 2 trang. |
| **Bảng ngang cực lớn vượt token limit** | Sub-group Split (xem 4.3) | Cắt theo nhóm phụ có sẵn (VD: Ngắn hạn / Dài hạn), giữ header + unit mỗi phần | Header bảng gốc, Đơn vị tính, Thuyết minh liên quan. |
| **Chính sách kế toán** *(các trang 23–42 văn bản)* | Policy Topic Level | Cắt theo mục chính sách (3.1 Tiền; 3.2 Hàng tồn kho) | Tên chính sách, Chuẩn mực (VAS/TT200). |
| **Document Summary** *(Chunk liên-note, xem 4.4)* | Document Level | 1 chunk duy nhất chứa chỉ mục tham chiếu chéo toàn báo cáo | Tên công ty, Năm TC, Danh sách tóm tắt từng note. |

### 4.3 Xử lý bảng cực lớn vượt giới hạn embedding model

> [!IMPORTANT]
> **Bổ sung từ bản V2:** Bản V1 quy tắc "giữ nguyên toàn bộ bảng trong 1 chunk" (Table-Atomic) đúng nguyên tắc, nhưng thiếu phương án cho bảng vượt giới hạn token. Ví dụ thực tế: BCTC có thuyết minh liệt kê 30–50 công ty liên quan có thể chứa bảng dài vượt giới hạn embedding model (BGE-M3: 8192 tokens, OpenAI `text-embedding-3-small`: 8191 tokens).

**Quy tắc xử lý:**
```text
IF total_tokens(table_chunk) <= EMBEDDING_TOKEN_LIMIT:
    → Giữ nguyên 1 chunk (Table-Atomic).
ELSE:
    → Tìm ranh giới cắt tự nhiên có sẵn trong chính bảng:
      1. Heading nhóm phụ: "Ngắn hạn" / "Dài hạn" (như TM13)
      2. Heading phân loại: "Nguyên giá" / "Khấu hao lũy kế" / "Giá trị còn lại" (như TM14)
      3. Nếu không có nhóm phụ → Cắt theo khoảng N dòng, đảm bảo:
         - Mỗi phần con bắt đầu bằng header row gốc + separator row
         - Mỗi phần con kèm dòng "[Đơn vị tính: triệu VND]"
         - KHÔNG CẮT giữa một dòng chỉ tiêu và dòng con thụt đầu dòng của nó

    → Mỗi phần con sinh ra 1 chunk riêng biệt với metadata:
      {
        "is_table_split": true,
        "split_index": 1,
        "split_total": 3,
        "parent_table_id": "note-13-table-summary"
      }
```

### 4.4 Chunk cấp tài liệu và tham chiếu chéo (Document Summary Chunk)

> [!IMPORTANT]
> **Bổ sung từ bản V2:** Bản V1 chỉ có chunk cấp note/statement riêng lẻ, thiếu hỗ trợ cho câu hỏi xuyên nhiều note (VD: *"So sánh tổng nợ vay với vốn chủ sở hữu 2 kỳ"*, hoặc câu hỏi cần đối chiếu giữa báo cáo chính và thuyết minh).

**Tạo 2 loại chunk cấp tài liệu:**

**① Document Summary Chunk:**
```text
chunk_type: "document_summary"

Nội dung: Tóm tắt cấu trúc toàn bộ báo cáo, gồm:
- Tổng tài sản: 432.xxx tỷ VND (từ Bảng cân đối)
- Tổng nợ: 282.xxx tỷ VND (từ Bảng cân đối)
- Tổng doanh thu thuần: xx.xxx tỷ VND (từ KQKD)
- Lợi nhuận sau thuế: xx.xxx tỷ VND (từ KQKD)
- Danh sách các thuyết minh: TM1 → TM40 kèm tên ngắn
- Số trang: 220 trang, kỳ báo cáo: FY2025
```

**② Cross-Reference Index Chunk:**
```text
chunk_type: "cross_reference_index"

Nội dung: Bảng ánh xạ giữa chỉ tiêu trên báo cáo chính ↔ thuyết minh chi tiết:
- "Hàng tồn kho" (dòng 141 BCĐKT) → Thuyết minh số 11 (trang 50)
- "Tài sản cố định hữu hình" (dòng 221 BCĐKT) → Thuyết minh số 14 (trang 53)
- "Vay và nợ thuê tài chính" (dòng 311 BCĐKT) → Thuyết minh số 20 (trang 58)
...
```

Khi retriever nhận được câu hỏi liên-note, nó sẽ match vào chunk `cross_reference_index` → biết cần kéo thêm chunk TM nào → retrieval 2 bước (two-hop retrieval).

---

## 5. Biểu diễn kép cho Chunk (Dual Representation)

Mỗi chunk lưu trữ 2 trường nội dung phục vụ 2 mục đích khác nhau:

### 5.1 Trường `content_retrieval` (Tối ưu cho Embedding & BM25)

Làm phẳng dữ liệu bảng và liên kết trực tiếp với chú thích thành các câu ngữ nghĩa tường minh:

```text
Báo cáo: Vinhomes (VHM) BCTC Hợp nhất 2025.
Thuyết minh 13: TÀI SẢN KHÁC (Đơn vị tính: triệu VND).
[Ngắn hạn]
- Đặt cọc cho mục đích đầu tư (i): Số cuối năm 88.791.672; Số đầu năm 25.626.349.
- Tài sản khác: Số cuối năm 81.044; Số đầu năm 217.425.
- TỔNG CỘNG Ngắn hạn: Số cuối năm 88.872.716; Số đầu năm 25.843.774.
  Trong đó: Đặt cọc bên khác 87.814.104; Đặt cọc bên liên quan (TM37) 1.058.612.
[Dài hạn]
- Đặt cọc cho mục đích đầu tư (ii): Số cuối năm 66.309.587; Số đầu năm 39.109.359.
- Đặt cọc cho mục đích thương mại (iii): Số cuối năm 1.032.336; Số đầu năm 1.032.336.
- Tài sản khác: Số cuối năm -; Số đầu năm 330.000.
- TỔNG CỘNG Dài hạn: Số cuối năm 67.341.923; Số đầu năm 40.471.695.
  Trong đó: Đặt cọc bên khác 13.939.923; Đặt cọc bên liên quan (TM37) 53.402.000.
[Chi tiết chú thích]
(i) Đặt cọc đầu tư ngắn hạn gồm: 12.000 tỷ VND giải phóng mặt bằng dự án BĐS
    (bảo đảm bằng cổ phiếu công ty trong Tập đoàn); 74.708 tỷ VND hợp tác phát
    triển và chuyển nhượng dự án tiềm năng; 2.084 tỷ VND chuyển nhượng cổ phần.
(ii) Đặt cọc đầu tư dài hạn gồm: 53.402 tỷ VND cho các công ty Tập đoàn và
     bên liên quan; 6.210 tỷ VND cho đối tác; 5.392 tỷ VND mua cổ phần;
     1.306 tỷ VND hợp đồng mua bán tài sản dự án.
(iii) Đặt cọc thương mại cho hợp đồng mua bán hàng hóa tương lai.
```

### 5.2 Trường `content_generation` (Tối ưu cho LLM Reasoning & Trích dẫn)

Chứa bảng Markdown hoặc HTML chuẩn nguyên bản kèm số liệu ma trận đầy đủ để LLM nhìn rõ cấu trúc dòng/cột, thực hiện đối chiếu chéo và sinh câu trả lời có kèm trích dẫn bảng biểu.

---

## 6. Schema Metadata Chuẩn Hóa Cho Vector DB (Milvus / Qdrant / Chroma)

> [!IMPORTANT]
> **Sửa lỗi từ bản V1:** Bản V1 không có trường QC/Evaluation nào trong metadata chunk. Hệ quả: chunk đã fail gate (VD: Numeric Precision < 99.5%) vẫn lọt vào Vector DB vì không có cơ chế filter ở tầng retrieval.
>
> Bản V2 bổ sung block `qc_*` fields kết nối trực tiếp với hệ thống evaluation đã thiết kế trong `README_parser_evaluation_v2.md` và `validate_financial_rules.py`, đảm bảo 2 tài liệu hoạt động như một hệ thống thống nhất.

```json
{
  "id": "VHM_2025_HN_NOTE_13",
  "document_id": "VHM_BCTC_2025_HopNhat",
  "ticker": "VHM",
  "fiscal_year": 2025,
  "period": "FY",
  "report_type": "consolidated",

  "chunk_type": "note_with_table",
  "section_code": "13",
  "section_title": "TÀI SẢN KHÁC",
  "parent_section": "THUYẾT MINH BÁO CÁO TÀI CHÍNH",
  "unit": {
    "raw": "triệu VND",
    "currency": "VND",
    "scale": "million"
  },
  "unit_raw": "triệu VND",
  "unit_currency": "VND",
  "unit_scale": "million",

  "pdf_page_range": [51, 52],
  "printed_page_range": [43, 44],
  "contains_tables": true,
  "table_names": ["Chi tiết tài sản khác ngắn hạn", "Chi tiết tài sản khác dài hạn"],

  "is_table_split": false,
  "split_index": null,
  "split_total": null,
  "parent_table_id": null,

  "stitched": true,
  "stitched_by_heuristic": false,
  "stitched_from_pages": [51, 52],

  "entity_mentions": [
    "Vingroup", "Thuyết minh số 37",
    "Dự án BĐS tiềm năng", "Giải phóng mặt bằng"
  ],

  "_comment_qc": "Các trường QC kết nối với validate_financial_rules.py và README_parser_evaluation_v2.md",
  "qc_status": "pass",
  "qc_numeric_precision": 1.0,
  "qc_semantic_association": 0.97,
  "qc_row_sum_pass_rate": 1.0,
  "qc_merge_cell_violations": 0,
  "qc_cross_run_stable": true,
  "qc_flagged_for_review": false,
  "qc_review_reason": null,

  "content_retrieval": "...",
  "content_generation": "..."
}
```

**Cơ chế hoạt động:**
1. Sau khi Parser V5 sinh chunk, chạy `validate_financial_rules.py` trên từng chunk chứa bảng.
2. Nếu bất kỳ gate nào fail (xem ngưỡng trong `README_parser_evaluation_v2.md`):
   - `qc_status` = `"fail"` hoặc `"warning"`
   - `qc_flagged_for_review` = `true`
   - `qc_review_reason` = mô tả cụ thể lỗi (VD: `"ROW_SUM fail: Giảm khác, delta=10"`)
3. Tại tầng retrieval, có thể filter: `WHERE qc_status != 'fail'` để loại chunk chất lượng kém. Hoặc linh hoạt hơn: vẫn trả về chunk fail nhưng kèm cảnh báo cho LLM biết dữ liệu có thể không chính xác.

---

## 7. Lộ Trình Triển Khai (Action Plan)

### 7.1 Bước 1 — Nâng cấp Parser Script (`parse_bctc_deepseek_v5.py`)
- Tích hợp prompt V5 với phân cấp heading chuẩn và cấm `&nbsp;`.
- Thêm `block_type` tagging cho header/footer/page_number trong canonical JSON.
- Bổ sung bộ sanitization (Mục 2.2) **với thứ tự rule đúng**.
- **Quan trọng:** Chưa triển khai đại trà. Chạy A/B test trước (Bước 4).

### 7.2 Bước 2 — Xây dựng Module Stitcher & Structure Chunker (`bctc_structure_chunker.py`)
- Viết logic ghép nối 3 tầng (Mục 4.1): mã số TM → vị trí liền kề → heuristic tiếp diễn.
- Xây dựng chunker phân loại 5 loại chunk (Mục 4.2).
- Thêm cơ chế xử lý bảng vượt token limit (Mục 4.3).
- Tạo generator biểu diễn kép (`content_retrieval` + `content_generation`).
- Sinh Document Summary và Cross-Reference Index chunks (Mục 4.4).

### 7.3 Bước 3 — Tích hợp QC Pipeline vào Metadata
- Chạy `validate_financial_rules.py` trên từng chunk chứa bảng sau khi sinh.
- Gắn `qc_*` fields vào metadata chunk (Mục 6).
- Đảm bảo chunk fail gate không lọt vào Vector DB (hoặc có cảnh báo kèm theo).

### 7.4 Bước 4 — A/B Test V4 vs V5 (BẮT BUỘC trước khi triển khai)

> [!WARNING]
> **Không bỏ qua bước này.** Prompt V5 thêm nhiều ràng buộc cấu trúc hơn V4. "Chặt chẽ hơn" KHÔNG đồng nghĩa "tốt hơn" — nếu model không tuân thủ đủ ràng buộc mới, kết quả có thể **tệ hơn V4** ở đúng loại trang rủi ro cao nhất.

> [!IMPORTANT]
> **Nguyên tắc phân định ranh giới Parser vs QC:**
> - **Parser KHÔNG thực hiện tính toán, không kiểm tra tổng.** Nhiệm vụ duy nhất của Parser là thị giác hóa chính xác tuyệt đối những gì in trên trang. Do đó, **tuyệt đối không dùng `ROW_SUM pass rate` làm metric đánh giá Parser A/B**. Nếu đưa row-sum vào evaluation của Parser, ta vô tình ép parser phải "suy luận/tính nhẩm", đi ngược lại nguyên tắc cấm reasoning.
> - **`ROW_SUM pass rate` và Financial Rules** thuộc phạm vi của **Post-Parser QC** (chạy sau khi Stitcher/Chunker đã hoàn tất, thông qua `validate_financial_rules.py`).

**Hệ thống Metric Đánh Giá Parser (Primary vs Secondary):**
- **Primary Metrics (Bắt buộc giữ vững):**
  1. *Numeric Accuracy*: Độ chính xác số liệu trích xuất (từng ký tự số, dấu chấm, dấu ngoặc).
  2. *Structural Accuracy*: Độ chính xác cấu trúc bảng (số cột, vị trí dòng header, phân tách ô).
  3. *Section / Heading Detection*: Nhận diện đúng mã và tiêu đề thuyết minh.
  4. *Table Integrity*: Bảng không bị vỡ cột, không bị gộp dòng sai lệch.
  5. *Continuation Signals*: Nhận diện đúng nhãn tiếp diễn / context candidates.
- **Secondary Metrics (Chất lượng định dạng):**
  1. *Formatting Compliance*: Tỷ lệ tuân thủ heading hierarchy (`#`, `##`, `###`), thẻ comment header/subheader.
  2. *Markdown Cleanliness*: Tỷ lệ triệt tiêu ký tự rác `&nbsp;`.

**Thiết kế thử nghiệm:**

| Thử nghiệm | Dữ liệu đầu vào | Đo lường (Parser A/B) |
|---|---|---|
| **Trang dễ** (văn bản thuyết minh đơn giản) | VHM p.52, MWG p.22 | Heading hierarchy compliance, `&nbsp;` elimination rate |
| **Trang trung bình** (bảng 3–5 cột, portrait) | VHM p.50–51, MWG p.23 | Numeric Accuracy, Table Integrity, `block_type` accuracy |
| **Trang khó** (bảng 7+ cột, landscape, auto-rotate) | VHM p.53–54, MWG p.27 | Numeric Accuracy, Table Integrity, Column alignment |
| **Trang rủi ro** (nhiều note trên 1 trang, tiếp diễn) | VHM p.51 (2 notes), MWG p.30 | Heading segmentation accuracy, `section_context_candidates` |

**Tiêu chí quyết định:**
- Ưu tiên tối thượng cho **Primary Metrics**. V5 không nhất thiết phải thắng V4 trên từng metric phụ (secondary), nhưng trên Primary Metrics (đặc biệt là Numeric Accuracy và Table Integrity) V5 không được phép suy giảm so với V4.
- Nếu V5 compliance rate < 90% trên heading hierarchy hoặc `block_type` tagging → xem xét giảm độ phức tạp prompt (loại bỏ bớt ràng buộc ít quan trọng để giữ vững độ chính xác số liệu).

### 7.5 Bước 5 — Kiểm định Retrieval End-to-End
- Chạy Retrieval Simulation 2 chế độ (Parser-only vs Full-pipeline) theo phương pháp đã thiết kế trong `README_parser_evaluation_v2.md mục 4.4`.
- Đo lường Recall@k, MRR trên tập câu hỏi `finance_eval_VHM.jsonl`.
- So sánh Structure Chunking vs Naive Chunking (fixed 512 tokens) vs Page-based Chunking.

---

## Phụ lục: Changelog V1 → V2

| # | Vấn đề V1 | Mức nghiêm trọng | Sửa trong V2 |
|---|---|---|---|
| 1 | Regex `Công ty Cổ phần` xóa nhầm dữ liệu thật | **Nghiêm trọng** | Thay bằng `block_type` tagging từ Parser (Mục 2.2) |
| 2 | Rule san phẳng `&nbsp;` chạy trước rule nhận diện indent → mất phân cấp | **Cao** | Đảo thứ tự: rule indent trước, rule san phẳng sau (Mục 2.2) |
| 3 | Số liệu ví dụ: 67.341.923 (Tổng DH) vs 66.309.587 (Sub-item) mô tả thiếu rõ ràng | **Trung bình** | Viết lại ví dụ tách rõ cấu trúc tổng/chi tiết (Mục 1.2, 5.1) |
| 4 | Stitcher chỉ match chuỗi tuyệt đối, không chịu được lỗi OCR | **Cao** | Cơ chế 3 tầng: mã số TM → vị trí liền kề → heuristic fallback (Mục 4.1) |
| 5 | Table-Atomic Chunk thiếu xử lý bảng vượt token limit | **Cao** | Sub-group split với header lặp lại mỗi phần (Mục 4.3) |
| 6 | Metadata schema tách rời hệ thống QC/Evaluation | **Trung bình** | Thêm block `qc_*` fields kết nối `validate_financial_rules.py` (Mục 6) |
| 7 | Thiếu chunk cấp tài liệu cho truy vấn liên-note | **Trung bình** | Thêm Document Summary + Cross-Reference Index chunks (Mục 4.4) |
| 8 | Action Plan thiếu A/B test V4 vs V5 | **Cao** | Thêm Bước 4 bắt buộc với tiêu chí quyết định cụ thể (Mục 7.4) |
