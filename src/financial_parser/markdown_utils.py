"""Loss-minimising Markdown helpers shared by engines, QC and canonical JSON."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, Tuple


NUMBER_PATTERN = re.compile(
    r"(?:(?<=(?<!\w)FY)|(?<![\w.]))"
    r"-?"
    r"(?:\d{1,3}(?:[,.]\d{3})+(?:[,.]\d+)?|\d+(?:[,.]\d+)?)"
    r"(?!\w|\.\d)"
)
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)
CLAUSE_BULLET_PATTERN = re.compile(
    r"(?:^|(?<=[\s;.,]))\(\s*\d{1,2}\s*\)(?=[ \t]+[A-Za-z]|\s*[:;])",
    re.MULTILINE,
)

CURRENCY_SYMBOLS = frozenset("$₫€£¥")

SCALE_WORDS_EN = frozenset({
    "million", "billion", "trillion", "thousand",
    "mn", "bn", "m", "b", "k",
})
SCALE_WORDS_VI = frozenset({
    "triệu", "tỷ", "tỉ", "nghìn", "ngàn", "trăm",
})
SCALE_WORDS = SCALE_WORDS_EN | SCALE_WORDS_VI

# Stopwords EN + VI dùng cho cross-page resolution (recall) và
# overflow exemption (precision).
STOPWORDS_EN = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "which",
    "who", "whom", "whose", "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "not", "no", "nor", "but", "if", "then", "than", "so",
})
STOPWORDS_VI = frozenset({
    "và", "của", "các", "có", "được", "cho", "trong", "về", "với",
    "những", "là", "tại", "theo", "đã", "từ", "này", "đó", "một",
    "hoặc", "như", "để", "khi", "thì", "mà", "ở", "ra", "vào", "lên",
    "xuống", "bởi", "do", "sẽ", "đang", "cũng", "rất", "quá", "nếu",
})
STOPWORDS = STOPWORDS_EN | STOPWORDS_VI


def is_resolvable_token(tok: str) -> bool:
    """Token được phép dùng để cross-page resolve / overflow exempt.

    Yêu cầu:
    - Không rỗng
    - Không phải stopword (EN + VI)
    - Chứa ít nhất 1 chữ cái (loại bỏ '--', '..', '%%', '§§', ...)
    """
    if not tok:
        return False
    if tok.lower() in STOPWORDS:
        return False
    return any(ch.isalpha() for ch in tok)


LEGAL_KEYWORDS = frozenset({
    "item", "section", "exhibit", "rule", "page", "note",
    "paragraph", "trang", "mục", "điều",
})
LEGAL_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in LEGAL_KEYWORDS) + r")\b"
    r"[\s:.\-]*$",   # keyword must be at end of prev_ctx (adjacent to number)
    re.IGNORECASE,
)

HEADER_COLLAPSED_VALUE_PATTERN = re.compile(
    r"^(?P<title>.*?)\s*[-–—]\s*(?P<val>[$₫€£¥]?\s*\(?[\d,]+(?:\.\d+)?\)?%?)$"
)

def _is_financial_cell_value(val: str) -> bool:
    v = val.strip()
    if any(c in v for c in ("$", "₫", "€", "£", "¥", "(", ")")):
        return True
    if "," in v:
        return True
    cleaned = re.sub(r"[^\d]", "", v)
    if cleaned and not (len(cleaned) == 4 and cleaned.startswith(("19", "20"))):
        return True
    return False

def decouple_collapsed_table_headers(text: str) -> str:
    """Decouple merged table headers where an engine collapsed the top data row into headers."""
    lines = text.splitlines()
    output_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            if is_table_separator(lines[i + 1]):
                header_cells = split_markdown_row(stripped)
                if len(header_cells) >= 2:
                    matched_items = []
                    for c in header_cells[1:]:
                        m = HEADER_COLLAPSED_VALUE_PATTERN.match(c)
                        if m and _is_financial_cell_value(m.group("val")):
                            matched_items.append(m)
                        else:
                            matched_items.append(None)

                    data_cols_count = len(header_cells) - 1
                    valid_matches = sum(1 for m in matched_items if m is not None)
                    if data_cols_count > 0 and valid_matches / data_cols_count >= 0.6:
                        new_headers = [""]
                        new_row_cells = [header_cells[0]]
                        for orig_c, m in zip(header_cells[1:], matched_items):
                            if m:
                                new_headers.append(m.group("title").strip())
                                new_row_cells.append(m.group("val").strip())
                            else:
                                new_headers.append(orig_c)
                                new_row_cells.append("")

                        output_lines.append("| " + " | ".join(new_headers) + " |")
                        output_lines.append(lines[i + 1])
                        output_lines.append("| " + " | ".join(new_row_cells) + " |")
                        i += 2
                        continue

        output_lines.append(line)
        i += 1
    return "\n".join(output_lines)


def sanitize_markdown(value: str) -> str:
    """Remove transport noise and clean duplicated merged table cells."""
    text = unicodedata.normalize("NFC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*```(?:markdown)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = clean_markdown_table_pipes(text.strip())
    text = decouple_collapsed_table_headers(text)
    return text.strip()


def split_markdown_row(row: str) -> List[str]:
    """Split a pipe row while respecting escaped literal pipes in cells."""
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells: List[str] = []
    current: List[str] = []
    escaped = False
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
            current.append(char)
        elif char == "|":
            cells.append("".join(current).replace("\\|", "|").strip())
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append("".join(current).replace("\\|", "|").strip())
    return cells


def is_table_separator(row: str) -> bool:
    if not row.strip().startswith("|"):
        return False
    cells = split_markdown_row(row)
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell or "") for cell in cells)


def deduplicate_merged_row(cells: List[str]) -> List[str]:
    """Deduplicate text in merged rows where Docling repeated the label across columns."""
    if len(cells) < 2:
        return cells
    first = cells[0].strip()
    raw_no_percent = first.rstrip("%").strip()
    if not first or NUMBER_PATTERN.fullmatch(first) or NUMBER_PATTERN.fullmatch(raw_no_percent) or first in ("-", "–", "—", "nil", "Nil", "N/A", "n/a"):
        return cells
    if all(cell.strip() == first for cell in cells[1:]):
        return [cells[0]] + ["" for _ in range(len(cells) - 1)]
    return cells


def clean_markdown_table_pipes(text: str) -> str:
    """Deduplicate repeated cells across columns in data rows of markdown pipe tables."""
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    in_table_data = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table_data = False
            cleaned_lines.append(line)
            continue
        # If the next line is a table separator, this line is the table header -> DO NOT DEDUP!
        if i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            in_table_data = False
            cleaned_lines.append(line)
            continue
        if is_table_separator(stripped):
            in_table_data = True
            cleaned_lines.append(line)
            continue
        if in_table_data:
            cells = split_markdown_row(stripped)
            deduped = deduplicate_merged_row(cells)
            if deduped != cells:
                cleaned_lines.append("| " + " | ".join(deduped) + " |")
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def extract_markdown_tables(markdown: str) -> List[dict]:
    """Read well-formed GitHub pipe tables with deduplicated merged cells."""
    lines = markdown.splitlines()
    tables: List[dict] = []
    index = 0
    while index + 1 < len(lines):
        if lines[index].strip().startswith("|") and is_table_separator(lines[index + 1]):
            header = split_markdown_row(lines[index])
            rows: List[List[str]] = []
            start_line = index + 1
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                raw_cells = split_markdown_row(lines[index])
                rows.append(deduplicate_merged_row(raw_cells))
                index += 1
            tables.append({"headers": header, "rows": rows, "start_line": start_line})
        else:
            index += 1
    return tables


def normalize_number_token(value: str) -> str:
    """Normalize number: remove thousand commas, currency symbols, and spaces."""
    token = value.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    negative = token.startswith("(") and token.endswith(")")
    for sym in CURRENCY_SYMBOLS:
        token = token.replace(sym, "")
    token = token.replace(",", "").strip()
    if token.startswith("- "):
        token = token[2:].strip()
    if negative and not token.startswith("-"):
        token = f"-{token.strip('()')}"
    return token


def number_multiset(text: str) -> Counter[str]:
    cleaned = CLAUSE_BULLET_PATTERN.sub(" ", text)
    return Counter(
        normalized
        for match in NUMBER_PATTERN.findall(cleaned)
        if (normalized := normalize_number_token(match))
    )


def _is_negative_sign(num: str) -> bool:
    """True nếu num bắt đầu bằng dấu âm thật."""
    return num.startswith("-")


def _has_scale_suffix(next_ctx: str) -> bool:
    """True nếu next_ctx bắt đầu bằng scale word (million, tỷ, ...)."""
    if not next_ctx:
        return False
    first = next_ctx.strip().split(maxsplit=1)[0].lower()
    first = first.rstrip(".,;:()[]")
    return first in SCALE_WORDS


def _has_legal_keyword(prev_ctx: str) -> bool:
    """True nếu prev_ctx kết thúc bằng legal keyword (Item, Section, ...)."""
    if not prev_ctx:
        return False
    return bool(LEGAL_KEYWORD_PATTERN.search(prev_ctx))


def _has_currency_before(prev_ctx: str) -> bool:
    """True nếu prev_ctx kết thúc bằng ký hiệu tiền tệ."""
    if not prev_ctx:
        return False
    return prev_ctx.rstrip()[-1:] in CURRENCY_SYMBOLS


def _has_percent_after(next_ctx: str) -> bool:
    """True nếu next_ctx bắt đầu bằng %."""
    if not next_ctx:
        return False
    return next_ctx.lstrip().startswith("%")


def _is_wrapped_in_parens(prev_ctx: str, next_ctx: str) -> bool:
    """True nếu số nằm trong ngoặc đơn (số âm kế toán)."""
    return (
        bool(prev_ctx) and prev_ctx.rstrip().endswith("(")
        and bool(next_ctx) and next_ctx.lstrip().startswith(")")
    )


def extract_numbers_with_context(
    text: str, window: int = 15
) -> List[Tuple[str, str, str, str]]:
    """Trích xuất số kèm context.

    Returns:
        List of (normalized_num, prev_ctx, next_ctx, raw_num).
        - normalized_num: "1234" từ "1,234"
        - prev_ctx: window ký tự trước số (stripped)
        - next_ctx: window ký tự sau số (stripped)
        - raw_num: chuỗi gốc "1,234"
    """
    cleaned = CLAUSE_BULLET_PATTERN.sub(" ", text)
    out: List[Tuple[str, str, str, str]] = []
    for m in NUMBER_PATTERN.finditer(cleaned):
        start, end = m.span()
        prev_ctx = cleaned[max(0, start - window):start].strip()
        next_ctx = cleaned[end:end + window].strip()
        raw = m.group(0)
        norm = normalize_number_token(raw)
        out.append((norm, prev_ctx, next_ctx, raw))
    return out


def classify_number(num: str, prev_ctx: str, next_ctx: str) -> str:
    """Phân loại số thành 'financial' hoặc 'metadata'.

    Args:
        num: chuỗi số THUẦN (có thể có dấu -, không có $ hay ngoặc).
             Ví dụ: "500", "1,234.56", "-150".
        prev_ctx: <= 15 ký tự ngay trước số (đã strip).
        next_ctx: <= 15 ký tự ngay sau số (đã strip).

    Returns: 'financial' | 'metadata'
    """
    if _has_currency_before(prev_ctx):
        return "financial"
    if _has_percent_after(next_ctx):
        return "financial"
    if _is_wrapped_in_parens(prev_ctx, next_ctx):
        return "financial"
    if _is_negative_sign(num):
        return "financial"
    if _has_scale_suffix(next_ctx):
        return "financial"
    if _has_legal_keyword(prev_ctx):
        return "metadata"
    if len(num) == 4 and num.isdigit() and 1900 <= int(num) <= 2099:
        return "metadata"
    if num.isdigit() and 0 <= int(num) <= 31:
        return "metadata"
    return "financial"


def token_multiset(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text)))


def multiset_recall(source: Counter[str], predicted: Counter[str]) -> float:
    if not source:
        return 1.0
    return sum((source & predicted).values()) / sum(source.values())


def multiset_precision(source: Counter[str], predicted: Counter[str]) -> float:
    if not predicted:
        return 1.0
    return sum((source & predicted).values()) / sum(predicted.values())
