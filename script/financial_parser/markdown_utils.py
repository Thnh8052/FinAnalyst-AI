"""Loss-minimising Markdown helpers shared by engines, QC and canonical JSON."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence


NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:\(?\s*[-−–—]?\s*(?:[$₫€£¥]\s*)?\d{1,3}(?:[,.]\d{3})+(?:[,.]\d+)?|\(?\s*[-−–—]?\s*(?:[$₫€£¥]\s*)?\d+(?:[,.]\d+)?)(?:\s*\)?)(?![\w.])"
)
TOKEN_PATTERN = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)


def sanitize_markdown(value: str) -> str:
    """Remove transport noise and clean duplicated merged table cells."""
    text = unicodedata.normalize("NFC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*```(?:markdown)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = clean_markdown_table_pipes(text.strip())
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
    """Canonical comparison key; keeps sign but ignores display separators/currency."""
    token = value.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    negative = token.startswith("(") and token.endswith(")")
    digits = re.sub(r"[^0-9]", "", token)
    if not digits:
        return ""
    return f"-{digits}" if negative or token.lstrip().startswith("-") else digits


def number_multiset(text: str) -> Counter[str]:
    return Counter(
        normalized
        for match in NUMBER_PATTERN.findall(text)
        if (normalized := normalize_number_token(match))
    )


def token_multiset(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text)))


def multiset_recall(source: Counter[str], predicted: Counter[str]) -> float:
    if not source:
        return 1.0
    return sum((source & predicted).values()) / sum(source.values())


def multiset_precision(source: Counter[str], predicted: Counter[str]) -> float:
    if not predicted:
        return 0.0
    return sum((source & predicted).values()) / sum(predicted.values())
