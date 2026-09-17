# Mental Health Chatbot → Vietnamese Financial Research Assistant
## Reference Architecture Mapping

## 0. Mục đích của tài liệu

Tài liệu này dùng repo **Bachelor_Thesis_Mental_Health_Chatbot** của bạn bạn như một **reference architecture** cho đề tài Finance.

Mục tiêu không phải copy repo, mà là:

- xác định thành phần nào có thể **REUSE** về mặt kiến trúc/tư duy;
- thành phần nào cần **ADAPT** vì domain Finance khác Mental Health;
- thành phần nào nên **REMOVE** vì không phục vụ bài toán Finance;
- thành phần nào phải **NEW** vì Finance có những yêu cầu mà Mental Health không có.

Định hướng Finance hiện tại:

> **Xây dựng hệ thống hỗ trợ tra cứu và phân tích thông tin tài chính doanh nghiệp Việt Nam sử dụng Retrieval-Augmented Generation**

Mục tiêu sản phẩm:

> **Bachelor thesis + mini production application**

---

# 1. Mental Health reference architecture

Repo Mental Health được xem như một hệ thống AI application khá đầy đủ, với các lớp chính:

```text
                       USER
                         |
                         v
                Input / Guardrails
                         |
                         v
                   Agent Router
                         |
              +----------+----------+
              |          |          |
              v          v          v
        Conversation    RAG       Web Search
                         |
                         v
                  Query Expansion
                         |
                         v
                 Hybrid Retrieval
                  Dense + BM25
                         |
                         v
                     Reranker
                         |
                         v
                 Context Selection
                         |
                         v
                    LLM Answer
                         |
                         v
                 Output / Guardrails
                         |
                         v
                       User

DATA SIDE:

PDF corpus / Web corpus
        |
        v
     Ingestion
        |
        v
 Validation / Staging
        |
        v
 Parsing / Chunking
        |
        v
 Embedding / Indexing
        |
        v
 Qdrant + document store

EVALUATION:

Retrieval evaluation
  baseline / subquery / rerank / both
        -> Recall@1/3/5, MRR

End-to-end RAG evaluation
        -> Faithfulness
        -> Answer Relevancy
        -> Context Precision
        -> Context Recall
```

Mental Health repo có thêm những application-specific modules như memory, Redis, medical/web sources, wellness, speech và human handoff. Những thứ này không tự động được chuyển sang Finance.

---

# 2. Mapping tổng quát

| Mental Health module | Finance tương ứng | Quyết định |
|---|---|---|
| PDF corpus | BCTC/BCTN/tài liệu tài chính | ADAPT |
| Web crawler | Public financial source connector | ADAPT |
| Source staging | Raw → validated → indexed financial sources | REUSE |
| Metadata | Company/period/document/source/metric metadata | ADAPT |
| Docling parsing | Docling/HTML/PDF financial parsing | REUSE + ADAPT |
| Image summarization | Có thể bỏ ở V0; chỉ dùng nếu figure/chart cần hiểu | REMOVE / LATER |
| Semantic chunking | Financial structure-aware chunking | ADAPT |
| Dense retrieval | Financial dense retrieval | REUSE |
| BM25 | Financial BM25 | REUSE |
| Hybrid retrieval | Dense + BM25 + fusion | REUSE |
| Cross-encoder reranker | Finance/Vietnamese-capable reranker | ADAPT |
| Sub-query generation | Financial query expansion/decomposition | ADAPT |
| Merge/dedupe/cap | Merge evidence từ nhiều retrieval path | REUSE |
| RAG agent | Finance tool/workflow agent | ADAPT |
| Web Search agent | Public financial source connector/retrieval | ADAPT |
| Medical knowledge base | Financial knowledge base | REMOVE / REPLACE |
| Wellness agent | Không cần | REMOVE |
| Medical-specific prompts | Financial prompts | REMOVE / REPLACE |
| Conversation agent | Financial conversation / clarification | ADAPT |
| Memory | Conversation/workspace context | ADAPT / LATER |
| Guardrails | Finance scope + unsupported-answer guardrails | ADAPT |
| Human handoff | Chưa cần ở core thesis | REMOVE / LATER |
| Speech | Không cần | REMOVE |
| Redis cache/session | Cache/session/jobs khi mini-production | REUSE / LATER |
| MongoDB/document DB | Application metadata/conversation/evaluation | ADAPT |
| Qdrant | Vector index | REUSE / ALTERNATIVE |
| RAGAS | Finance answer evaluation | ADAPT |
| Retrieval eval | Finance retrieval benchmark | REUSE |
| Numerical verification | Không có tương đương trực tiếp | NEW |
| Structured financial data | XBRL/API/provider data | NEW |
| Calculator | Deterministic financial calculation | NEW |
| Evidence/citation verification | Claim ↔ source/evidence checking | NEW |
| Financial metric normalization | Revenue/profit/assets/etc. semantics | NEW |
| Reporting-period normalization | Year/quarter/unit/currency handling | NEW |

---

# 3. REUSE — Những gì có thể giữ gần như nguyên về mặt tư duy

## 3.1. Tách Data Pipeline khỏi AI Pipeline

Mental Health cho thấy data ingestion/processing/indexing có thể tách khỏi query-time RAG.

Finance nên giữ:

```text
DATA TIME

source
 -> ingest
 -> validate
 -> parse
 -> chunk
 -> enrich
 -> index

QUERY TIME

question
 -> understand
 -> retrieve
 -> rerank
 -> reason
 -> generate
 -> verify
```

Đây là REUSE quan trọng nhất.

---

## 3.2. Staging / validation trước indexing

Mental Health có cách tổ chức source theo trạng thái trước khi index.

Finance có thể áp dụng:

```text
RAW
  |
  v
VALIDATING
  |
  +---- invalid --> REJECTED
  |
  v
APPROVED
  |
  v
PROCESSED
  |
  v
INDEXED
```

Mục đích:

- không index document lỗi;
- theo dõi nguồn nào đã xử lý;
- dễ reprocess;
- reproducibility tốt hơn.

---

## 3.3. Dense + BM25

Mental Health sử dụng cả semantic và lexical retrieval.

Finance giữ nguyên pattern:

```text
                     query
                       |
              +--------+--------+
              |                 |
              v                 v
           Dense             BM25
              |                 |
              +--------+--------+
                       |
                     Fusion
                       |
                       v
                    Rerank
```

Không cần chứng minh rằng dense hoặc BM25 phải tốt hơn trước; evaluation sẽ quyết định.

---

## 3.4. Reranking là stage thứ hai

Giữ nguyên tư duy:

```text
Retriever = high recall
Reranker  = improve ranking
```

Không đưa reranker vào V0 nếu mục tiêu là tạo baseline đơn giản.

Sau đó làm experiment:

```text
Dense
vs
Dense + BM25
vs
Dense + BM25 + reranker
```

---

## 3.5. Sub-query / query expansion

Mental Health có evaluation riêng cho subquery.

Finance có thể áp dụng cho các câu phức tạp:

```text
"Doanh thu và lợi nhuận của FPT thay đổi thế nào từ 2024 đến 2025?"
```

có thể tách thành:

```text
Q1: FPT revenue 2024
Q2: FPT revenue 2025
Q3: FPT net profit 2024
Q4: FPT net profit 2025
```

Sau đó merge/dedupe/rerank.

---

## 3.6. Merge / dedupe / context cap

Mental Health không đưa mọi retrieved result vào LLM.

Finance giữ pattern:

```text
retrieval results
 -> merge
 -> dedupe
 -> select top-N
 -> context
```

Điều này giúp kiểm soát context size và tránh duplicate evidence.

---

## 3.7. Component-level retrieval evaluation

Đây là một phần nên REUSE gần như nguyên tư duy.

Ví dụ:

```text
Experiment 0: Dense
Experiment 1: BM25
Experiment 2: Hybrid
Experiment 3: Hybrid + Reranker
Experiment 4: Hybrid + Reranker + Query Expansion
```

Metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
Precision@K
nDCG@K (optional)
```

---

## 3.8. End-to-end RAG evaluation

Mental Health dùng RAGAS cho:

```text
Faithfulness
Answer Relevancy
Context Precision
Context Recall
```

Finance có thể REUSE framework, nhưng evaluation set và rubric phải là Finance-specific.

---

# 4. ADAPT — Những gì giữ kiến trúc nhưng phải đổi theo Finance

# 4.1. Corpus

Mental Health:

```text
medical PDFs
medical web sources
```

Finance:

```text
Vietnamese BCTC
Vietnamese BCTN
public financial documents
user-provided financial documents
structured financial data
```

Không nên coi tất cả source là cùng một loại.

---

# 4.2. Crawler

Mental Health crawler hướng đến medical/web sources.

Finance cần:

```text
Public financial source adapter
```

thay vì generic web crawling.

Mục tiêu:

```text
discover document
 -> obtain metadata
 -> download/fetch
 -> validate
 -> process
```

Ưu tiên nguồn tài chính đáng tin cậy; không crawl Internet đại trà.

---

# 4.3. Metadata

Mental Health metadata có publisher/language/trust/relevance/content type...

Finance cần thêm:

```text
company
company_id / ticker
reporting_period
publication_date
document_type
statement_type
currency
unit
consolidated_or_standalone
source
source_url
page
section
```

Đây là phần cực kỳ quan trọng để retrieval chính xác.

---

# 4.4. Chunking

Mental Health semantic/section chunking có thể được giữ concept.

Finance phải điều chỉnh để tôn trọng:

```text
financial report structure
section
subsection
table
footnote
page
metric
reporting period
```

Ví dụ không nên phá một table thành các text fragments không còn quan hệ dòng/cột.

---

# 4.5. Reranker

Mental Health reranker là multilingual/medical-oriented implementation.

Finance không nhất thiết dùng đúng model đó.

Ta benchmark:

```text
pretrained multilingual reranker
vs
Vietnamese-capable reranker
vs
financial-domain reranker nếu phù hợp
```

Chỉ giữ model tốt nhất theo evaluation.

---

# 4.6. Conversation / clarification

Mental Health có conversation routing.

Finance giữ khả năng conversation nhưng đổi mục tiêu:

```text
User:
"Doanh thu thế nào?"

Assistant:
"Bạn muốn xem doanh thu của công ty nào và giai đoạn nào?"
```

Clarification là supporting capability.

---

# 4.7. Agent router

Mental Health router:

```text
Conversation
RAG
Web
Wellness
```

Finance nên đơn giản hơn:

```text
Document RAG
Structured Data
Calculator
Clarification
Verification
```

Một query có thể gọi nhiều tool.

Ví dụ:

```text
"Doanh thu tăng bao nhiêu và tại sao?"

Structured Data
      +
Calculator
      +
Document RAG
```

---

# 4.8. Guardrails

Medical safety guardrails không chuyển nguyên xi.

Finance cần guardrails cho:

```text
scope control
unsupported financial claims
investment-advice boundary
missing evidence
conflicting data
```

Ví dụ:

```text
"Tôi nên mua cổ phiếu X không?"
```

không thuộc core use case của hệ thống.

---

# 4.9. Memory

Không cần copy long-term medical memory.

Finance V0 chỉ cần:

```text
recent conversation
current company
current period
current workspace
```

Long-term user memory để LATER.

---

# 5. REMOVE — Những thành phần của Mental Health không nên đưa vào Finance core

## 5.1. Wellness agent

Không có tương đương trong Finance.

REMOVE.

---

## 5.2. Medical-specific knowledge / medical prompts

REMOVE.

Thay bằng financial prompt/domain policy.

---

## 5.3. Medical external APIs

Ví dụ PubMed/Europe PMC và các medical search flow.

REMOVE.

Finance dùng financial data connectors.

---

## 5.4. Speech pipeline

Không cần nếu use case hiện tại là financial research assistant.

REMOVE khỏi thesis core.

---

## 5.5. Human handoff

Không cần trong baseline.

Có thể đưa vào future work nếu sau này muốn mở thành analyst escalation workflow.

---

## 5.6. Medical web search fallback

Không copy nguyên dạng.

Nếu Finance cần fallback thì phải chuyển thành:

```text
trusted financial source connector
```

chứ không phải generic web search.

---

# 6. NEW — Những thứ Finance cần mà Mental Health không cung cấp đủ

Đây là phần quan trọng nhất.

# 6.1. Structured Financial Data Layer

Finance cần một nhánh knowledge riêng:

```text
Financial API / structured dataset
        |
        v
normalize
        |
        v
financial facts
```

Ví dụ:

```text
company
metric
period
value
unit
currency
source
```

Nó khác Document RAG.

---

# 6.2. Financial Metric Normalization

Hệ thống cần hiểu các khái niệm tài chính tương ứng.

Ví dụ:

```text
Revenue
Net Revenue
Doanh thu
Doanh thu thuần
```

cần được map thích hợp tùy data source.

Tương tự:

```text
Net Income
Profit After Tax
Lợi nhuận sau thuế
```

Đây là Finance-specific data semantics.

---

# 6.3. Period / Unit / Currency Handling

Ví dụ cùng một metric phải phân biệt:

```text
FY2024
Q4 2024
9M 2024
2024 YTD
```

và:

```text
VND
million VND
billion VND
USD
```

Nếu không chuẩn hóa, retrieval có thể đúng document nhưng answer vẫn sai.

---

# 6.4. Deterministic Financial Calculator

Mental Health không cần component tương đương.

Finance cần:

```text
LLM
 -> identifies required calculation
 -> calculator executes
 -> result returned
```

Ví dụ:

```text
Revenue Growth
= (new - old) / old
```

Có thể mở rộng:

```text
Gross Margin
Operating Margin
ROA
ROE
Debt-to-Equity
...
```

---

# 6.5. Numerical Verification

Finance có thể kiểm tra deterministic:

```text
LLM result
vs
calculator result
```

Đây là một điểm reliability rất mạnh của domain Finance.

---

# 6.6. Evidence / Claim Verification

Mental Health repo có RAGAS, nhưng Finance có thể đi xa hơn:

```text
Answer
 -> claims
 -> evidence mapping
 -> supported / unsupported
```

Ví dụ:

```text
Claim 1 -> BCTC page 42 ✓
Claim 2 -> BCTN page 71 ✓
Claim 3 -> no supporting evidence ✗
```

Claim không được support thì:

```text
rewrite
or
abstain
```

---

# 6.7. Source-aware financial answer

Answer cần lưu traceability:

```text
claim
source
page
section
metric
period
```

Để user có thể kiểm tra nguồn.

---

# 6.8. Conflicting-source handling

Finance có thể gặp:

```text
Provider A = X
Provider B = Y
PDF = Z
```

V0 có thể chỉ flag conflict.

Sau này mới làm reconciliation.

---

# 7. Finance architecture sau khi mapping

Sau khi loại bỏ những thành phần không phù hợp và bổ sung phần Finance-specific, kiến trúc nên là:

```text
                     USER
                       |
                       v
                Financial Assistant
                       |
                       v
              Query Understanding
                       |
                       v
                Finance Router
                       |
         +-------------+-------------+
         |             |             |
         v             v             v
     Clarify       Document RAG   Structured Data
                         |             |
                         v             v
                    Dense + BM25    Lookup
                         |
                         v             |
                      Reranker        |
                         |             |
                         +------+------+
                                |
                                v
                         Reasoning Layer
                                |
                         +------+------+
                         |             |
                         v             v
                    Calculator     Evidence
                         |         Selection
                         |             |
                         +------+------+
                                |
                                v
                           LLM Generate
                                |
                                v
                           Verification
                                |
                       +--------+--------+
                       |                 |
                       v                 v
                     PASS         INSUFFICIENT
                       |                 |
                       v                 v
                Answer + Source     Abstain / Ask
                       |
                       v
                   Evaluation
```

DATA SIDE:

```text
              DATA SOURCES
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
   User Files   Public Docs   Structured Data
        |           |           |
        +-----------+-----------+
                    |
                    v
               INGESTION
                    |
                    v
              VALIDATION
                    |
                    v
            DOCUMENT PROCESSING
                    |
             +------+------+
             |             |
             v             v
          Text/        Tables / Facts
         Structure          |
             |              |
             v              v
       Chunk + Enrich   Normalize
             |              |
             v              v
        Dense/BM25      Structured DB
```

---

# 8. V0 — Những gì Finance cần xây trước

Không lấy toàn bộ Mental Health architecture.

V0 chỉ cần:

```text
User upload
   |
   v
Parse document
   |
   v
Structure-aware chunking
   |
   v
Embedding
   |
   v
Vector retrieval
   |
   v
LLM
   |
   v
Answer + source
```

Mục tiêu của V0:

> Có baseline đơn giản, reproducible và evaluate được.

---

# 9. V1 — Retrieval improvement

```text
V0
 |
 +-- BM25
 |
 +-- Hybrid fusion
 |
 +-- Reranker
 |
 +-- Retrieval evaluation
```

Experiment:

```text
Dense
vs
Dense + BM25
vs
Dense + BM25 + Reranker
```

Đây là phần lấy trực tiếp nhiều nhất từ Mental Health evaluation design.

---

# 10. V2 — Finance capability

```text
V1
 |
 +-- structured financial data
 |
 +-- metric normalization
 +-- period/unit normalization
 +-- calculator
```

Use case:

```text
Revenue 2024
Revenue 2025
       |
       v
Calculator
       |
       v
Growth %
```

---

# 11. V3 — Agent / Conversation

```text
V2
 |
 +-- query understanding
 +-- clarification
 +-- tool routing
 +-- conversation context
```

Router:

```text
Question
 |
 +-- document?
 +-- structured data?
 +-- calculator?
 +-- clarification?
```

---

# 12. V4 — Reliability

```text
V3
 |
 +-- evidence verification
 +-- numerical verification
 +-- citation/source validation
 +-- abstention
```

Flow:

```text
Answer
  |
  v
Verify claims
  |
  +---- supported ----> Answer
  |
  +---- unsupported ---> Rewrite / Abstain
```

---

# 13. V5 — Mini Production

```text
V4
 |
 +-- public-source automation
 +-- user workspace
 +-- asynchronous ingestion
 +-- Redis
 +-- persistent database
 +-- logging
 +-- monitoring
 +-- evaluation dashboard
 +-- Docker/deployment
```

Redis should be used for:

```text
cache
session/temporary state
job queue/job state
rate limiting
```

It is infrastructure, not part of the RAG algorithm.

---

# 14. What we should NOT copy from Mental Health

Do not copy architecture just because it exists in the reference repo.

```text
❌ Wellness agent
❌ Medical APIs
❌ Speech pipeline
❌ Medical web search logic
❌ Human handoff
❌ Long-term medical memory
❌ Multi-agent complexity without a measured need
❌ Every infrastructure component from day one
```

---

# 15. What we SHOULD copy conceptually

```text
✓ Data pipeline separated from query pipeline
✓ Source validation/staging
✓ Metadata-first ingestion
✓ Hybrid retrieval
✓ Reranking
✓ Sub-query experimentation
✓ Merge/dedupe/cap
✓ Retrieval-level evaluation
✓ End-to-end RAG evaluation
✓ Evaluation artifacts saved separately
✓ Production-oriented failure/fallback thinking
✓ Modular backend architecture
```

---

# 16. What makes Finance meaningfully different

The biggest difference is that Finance requires two knowledge modes:

```text
                  FINANCIAL KNOWLEDGE
                         |
              +----------+----------+
              |                     |
              v                     v
       Unstructured docs       Structured facts
              |                     |
          Document RAG          Data lookup
              |                     |
              +----------+----------+
                         |
                         v
                    Reasoning
                         |
                    Calculator
                         |
                         v
                    Verification
```

Mental Health reference architecture mainly demonstrates the first side:

```text
Documents -> RAG -> Agent -> Answer
```

Finance expands this to:

```text
Documents + Structured Financial Facts
          -> RAG / Tools
          -> Reason / Calculate
          -> Verify
          -> Answer
```

This is the core architectural difference.

---

# 17. Final decision: what exactly does the Finance project need?

## MUST BUILD

```text
1. Financial data/document ingestion
2. Document processing
3. Structure-aware chunking + metadata
4. Dense retrieval
5. BM25 / hybrid retrieval
6. Reranking
7. Financial structured-data access
8. Deterministic calculator
9. Financial reasoning
10. Evidence/source grounding
11. Evaluation
12. Basic web application
```

## SHOULD BUILD

```text
13. Query understanding
14. Clarification
15. Tool routing / lightweight agent
16. Numerical verification
17. Citation/evidence verification
18. Abstention
```

## MINI-PRODUCTION

```text
19. Automated public-source ingestion
20. Workspace
21. Async jobs
22. Redis
23. Persistent DB
24. Logging/monitoring
25. Docker/deployment
```

## OPTIONAL / FUTURE

```text
26. VLM document fallback
27. Domain-specific fine-tuning
28. Financial knowledge graph
29. Multi-agent system
30. Long-term memory
31. Multi-country support
32. Real-time market intelligence
```

---

# 18. Final Reuse / Adapt / Remove / New summary

```text
                      MENTAL HEALTH
                           |
          +----------------+----------------+
          |                |                |
       REUSE             ADAPT           REMOVE
          |                |                |
   RAG architecture     Corpus          Wellness
   Hybrid retrieval     Metadata        Speech
   BM25/Dense           Chunking        Medical APIs
   Reranking            Reranker        Medical logic
   Sub-query            Router          Human handoff
   Evaluation           Guardrails
   Staging              Memory
          |                |
          +----------------+
                   |
                   v
                FINANCE
                   |
                  NEW
                   |
        Structured financial data
        Metric normalization
        Period/unit handling
        Calculator
        Numerical verification
        Evidence verification
        Financial source handling
```

---

# 19. Bottom line

Mental Health repo nên được xem là **reference architecture**, không phải implementation blueprint.

Nó cho ta cách tổ chức một AI application thật:

```text
Data
 -> Processing
 -> Knowledge
 -> Retrieval
 -> Reranking
 -> Agent
 -> Generation
 -> Evaluation
 -> Production infrastructure
```

Finance giữ phần khung đó nhưng thay domain logic và thêm các capability đặc thù:

```text
Financial documents
+
Structured financial data
+
Financial reasoning
+
Calculator
+
Evidence verification
```

Vì vậy project cuối cùng không phải là:

> “Copy Mental Health Chatbot nhưng đổi medical thành finance.”

Mà là:

> **“Dùng một architecture đã được kiểm chứng ở một Bachelor project thực tế làm reference, sau đó thiết kế lại knowledge/reasoning/reliability layer cho bài toán Financial Research.”**

Đây là distinction quan trọng nếu sau này phải giải thích với giảng viên tại sao các component của hệ thống tồn tại.
