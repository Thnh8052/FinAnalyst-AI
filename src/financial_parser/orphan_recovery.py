"""Orphan text recovery for Docling TableFormer dropped cells and collapsed tables.

Follows the verified recovery architecture:
1. Reconstructs cross-page table continuations where trailing rows overflowed from the previous page,
   inheriting column headers to preserve structured Markdown tables.
2. Reconstructs degenerate/collapsed tables (e.g. legal exhibits or wall of text swallowed into header)
   into clean structured text blocks.
3. Identifies truly missing text blocks using Sub-block Splitting, 3-Point Anchor Matching
   (Head, Mid, Tail), and Cross-Page Context Awareness ([N-1, N, N+1]), avoiding artificial duplication.
4. Cleans font artifacts (e.g. \\ufffd replacement chars) and applies positional insertion.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from .markdown_utils import extract_markdown_tables

HEADER_KEYWORDS = {
    "year ended", "years ended", "three months ended", "six months ended",
    "nine months ended", "twelve months ended", "quarter ended", "as of",
    "at december", "at september", "at march", "at june",
    "target", "actual", "budget", "forecast", "variance",
    "estimate", "projected", "plan", "description", "category", "jurisdiction",
    "classification", "q1", "q2", "q3", "q4", "fy", "fy15", "fy16", "fy17",
}


ENTITY_SUFFIXES = {
    "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "s.a.", "s.a", "bv", "b.v.", "srl", "gmbh", "co.", "co"
}


class OrphanTextRecoverer:
    """Recovers text and numbers dropped by Docling table structure recognition with cross-page context awareness."""

    def __init__(
        self,
        min_char_len: int = 25,
        huge_cell_threshold: int = 500,
        margin_band: float = 40.0,
    ) -> None:
        self.min_char_len = min_char_len
        self.huge_cell_threshold = huge_cell_threshold
        self.margin_band = margin_band

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text to lowercase alphanumeric with single spaces for robust comparison."""
        return re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def clean_pymupdf_text(text: str) -> str:
        """Cleans \ufffd encoding artifacts frequently produced by PyMuPDF on SEC filings."""
        cleaned = re.sub(r"(\d)\s*\ufffd\s*(\d)", r"\1 - \2", text)
        cleaned = re.sub(r"([a-zA-Z])\ufffd([a-zA-Z])", r"\1'\2", cleaned)
        cleaned = cleaned.replace("\ufffd", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def check_presence(self, sub_p: str, corpus_norm: str) -> str:
        """Determines whether a paragraph or sub-block is already present in the corpus.

        Uses exact substring match for short phrases (< 5 words) and 3-point anchor matching
        (Head, Mid, Tail) for longer paragraphs. This resolves 'abc' vs 'abcdef' substring cases
        and protects against word-bag collisions in SEC filings.
        """
        p_norm = self._normalize(sub_p)
        words = p_norm.split()
        if not words:
            return "present"

        # Case 1: Short text or exact substring match (e.g. 'abc' in 'abcdef')
        if len(words) < 5 or p_norm in corpus_norm:
            return "present" if p_norm in corpus_norm else "missing"

        # Case 2: Three-Point Anchor Matching (Head, Mid, Tail)
        head = " ".join(words[:min(5, len(words))])
        tail = " ".join(words[-min(5, len(words)):])
        mid_idx = len(words) // 2
        mid = " ".join(words[mid_idx : mid_idx + min(5, len(words) - mid_idx)])

        has_head = head in corpus_norm
        has_tail = tail in corpus_norm
        has_mid = mid in corpus_norm

        # Majority anchor match (at least 2 of 3 anchors match).
        # This handles minor floating glyphs, checkboxes (YES/NO), or trailing line-wraps.
        matches = sum([has_head, has_mid, has_tail])
        if matches >= 2:
            return "present"

        return "missing"

    def is_degenerate_table(
        self,
        tables: List[dict],
        markdown: str,
        raw_text: str,
        char_count: int,
    ) -> bool:
        """Detect if Docling falsely forced narrative text into a broken/mangled table."""
        if not tables:
            return False

        total_data_rows = sum(len(t["rows"]) for t in tables)

        # Case 1: Table has headers but 0 data rows (table header swallowed content)
        if total_data_rows == 0:
            return True

        # Case 2: Only 1 small table with <= 1 data row, but page has substantial text (> 500 chars)
        if len(tables) == 1 and total_data_rows <= 1 and char_count > 500:
            # Check for a huge wall of text shoved into a single cell
            for table in tables:
                for row in table["rows"]:
                    for cell in row:
                        if len(cell) >= self.huge_cell_threshold:
                            return True

            # Check vocabulary overlap between markdown and raw text
            md_clean = re.sub(r"[^a-zA-Z0-9]+", " ", markdown.lower())
            md_words = set(md_clean.split())
            raw_words = re.findall(r"[a-zA-Z0-9]+", raw_text.lower())
            if raw_words:
                overlap = sum(1 for w in raw_words if w in md_words) / len(raw_words)
                if overlap < 0.70:
                    return True

        return False

    @staticmethod
    def is_genuine_table_header(cells: List[str]) -> bool:
        """A genuine table header has descriptive column labels, NOT numeric data values in non-label columns."""
        if not cells or len(cells) < 2:
            return False
        non_first_cells = [c.strip() for c in cells[1:] if c.strip()]
        if not non_first_cells:
            return False

        # Financial header case: all non-first cells are 4-digit calendar years or quarters
        # e.g. ['(Dollars in millions)', '2015', '2014', '2013'] or ['December 31,', '2016', '2015']
        if all(
            re.match(r"^(?:19|20)\d{2}$", c) or re.match(r"^q[1-4](?:\s*(?:19|20)\d{2})?$", c.lower())
            for c in non_first_cells
        ):
            return True

        # If >= 50% of non-first columns are numeric / currency / dash, it is a data row misclassified by Docling
        num_numeric = sum(
            1 for c in non_first_cells
            if not c or c in ["-", "—", "–", "N/A", "n/a"] or re.match(r"^[$₫€£¥]?\s*\(?[\d,.\s]+%?\)?$", c)
        )
        if (num_numeric / len(non_first_cells)) >= 0.5:
            return False
        return True

    @classmethod
    def is_valid_inherited_header(cls, headers: List[str]) -> bool:
        """Verifies that candidate headers from previous page are genuine column headers, not a swallowed data row."""
        if not headers or len(headers) < 2:
            return False

        first_cell = headers[0].strip()
        words = [w.lower().rstrip(".,") for w in first_cell.split()]
        for w in words:
            if w in ENTITY_SUFFIXES:
                return False

        # Headers must NOT contain numeric currency values in data columns
        for h in headers[1:]:
            if re.match(r"^[$₫€£¥]\s*\(?[\d,.\s]+%?\)?$", h.strip()):
                return False

        # Check if first cell looks like a line item narrative sentence
        if len(first_cell) > 50 and any(
            w in first_cell.lower()
            for w in ["timing of", "increase in", "decrease in", "lower operating", "higher vat"]
        ):
            return False

        # For 2-column tables, require explicit financial/reporting header cues and short labels
        if len(headers) == 2:
            if any(len(h.strip()) > 35 for h in headers):
                return False
            h_text = " ".join(headers).lower()
            financial_cues = ["year", "period", "date", "$", "dollar", "amount", "total", "201", "202"]
            if not any(cue in h_text for cue in financial_cues):
                return False

        return True

    @staticmethod
    def is_table_header_row(cells: List[str]) -> bool:
        """Deterministic check if a row is a table header rather than a data row.
        
        Guards against false continuation:
        1. Contains reporting period / budget / variance header keywords.
        2. First column is a standalone 4-digit year or quarter label.
        """
        if not cells:
            return False

        row_text = " ".join(cells).lower()
        for kw in HEADER_KEYWORDS:
            if kw in row_text:
                return True

        first_cell = cells[0].strip()
        if re.match(r"^(?:19|20)\d{2}$", first_cell) or re.match(r"^q[1-4]$", first_cell.lower()):
            return True

        return False

    @classmethod
    def is_valid_continuation_data_row(cls, cells: List[str], expected_cols: int) -> bool:
        """Verifies that a candidate row is a genuine table continuation data row.
        
        Criteria:
        1. Exact column count match.
        2. Must NOT be an independent table header row.
        3. Column 0 is a row label (not empty, not pure year, not decorative divider).
        """
        if len(cells) != expected_cols:
            return False

        if cls.is_table_header_row(cells) or cls.is_genuine_table_header(cells):
            return False

        first_cell = cells[0].strip()
        if not first_cell or first_cell == "_____________________________":
            return False

        return True

    @staticmethod
    def is_table_at_bottom_of_markdown(markdown: str) -> bool:
        """Check if the last table in markdown is actually at the bottom of the page."""
        tables = extract_markdown_tables(markdown)
        if not tables:
            return False
        last_t = tables[-1]
        lines = markdown.splitlines()
        start = max(0, last_t.get("start_line", 1) - 1)
        end_idx = start + 2 + len(last_t.get("rows", []))
        after_lines = lines[end_idx:]
        after_narrative = " ".join(
            l.strip() for l in after_lines
            if l.strip() and not l.strip().startswith("|") and not l.strip().startswith("#") and len(l.strip()) > 30
        )
        return len(after_narrative) < 100

    @classmethod
    def extract_top_table_rows(
        cls, page: Any, max_y: float = 120.0, num_cols: int = 0
    ) -> Tuple[List[List[str]], int]:
        """Extract consecutive horizontal rows from the top band of the page matching num_cols."""
        try:
            td = page.get_text("dict")
        except Exception:
            return [], 0

        blocks = [b for b in td.get("blocks", []) if b.get("type") == 0]
        if not blocks:
            return [], 0

        extracted_rows: List[List[str]] = []
        last_block_idx = 0

        for idx, b in enumerate(blocks):
            bbox = b.get("bbox", [0, 0, 0, 0])
            if bbox[1] > max_y:
                break

            lines = b.get("lines", [])
            if not lines:
                continue

            items: List[Tuple[float, str]] = []
            for l in lines:
                t = "".join(s.get("text", "") for s in l.get("spans", [])).strip()
                if t and t != "_____________________________":
                    items.append((l.get("bbox", [0, 0, 0, 0])[0], t))

            if not items:
                continue

            items.sort(key=lambda x: x[0])

            cells: List[str] = []
            i = 0
            while i < len(items):
                x, text = items[i]
                if text in ["$", "€", "£", "¥"] and i + 1 < len(items):
                    cells.append(f"{text} {items[i+1][1]}")
                    i += 2
                else:
                    cells.append(text)
                    i += 1

            if len(cells) == num_cols and cls.is_valid_continuation_data_row(cells, num_cols):
                extracted_rows.append(cells)
                last_block_idx = idx + 1
            else:
                break

        return extracted_rows, last_block_idx

    def try_reconstruct_table_continuation(
        self,
        page: Any,
        prev_markdown: Optional[str],
        current_markdown: str = "",
    ) -> Optional[Tuple[str, str]]:
        """If page starts with trailing table row(s) continuing the last table of prev_page,
        reconstructs a complete Markdown Table with inherited headers.
        Protected by 4 guardrails against misinheritance:
        1. Current page does not already have a valid multi-row table.
        2. Previous page table was at the bottom of the page.
        3. Row cells are validated financial data cells (not an independent table header).
        4. Supports multi-row overflow tables.
        """
        if not prev_markdown:
            return None

        # Guard 2: Table on previous page must be at the very bottom
        if not self.is_table_at_bottom_of_markdown(prev_markdown):
            return None

        prev_tables = extract_markdown_tables(prev_markdown)
        if not prev_tables:
            return None

        last_table = prev_tables[-1]
        headers = list(last_table.get("headers", []))
        num_cols = len(headers)
        if num_cols < 2:
            return None

        # Multi-level header clean
        if all(not h.strip() for h in headers) or sum(1 for h in headers if h.strip()) < num_cols // 2:
            candidate_rows = [last_table.get("headers", [])] + last_table.get("rows", [])[:3]
            best_r = max(
                candidate_rows,
                key=lambda r: sum(1 for c in r if c.strip() and not re.match(r"^col_\d+$", c.lower())),
            )
            if sum(1 for c in best_r if c.strip()) > sum(1 for h in headers if h.strip()):
                headers = [c.strip() if c.strip() else f"Col_{idx+1}" for idx, c in enumerate(best_r)]

        # Guard 2b: Inherited headers must be valid, genuine headers (not an entity or line item amount)
        if not self.is_valid_inherited_header(headers):
            return None

        # Guard 1: Check if current page already has a table with >= 2 data rows
        if current_markdown:
            curr_tables = extract_markdown_tables(current_markdown)
            if curr_tables and sum(len(t.get("rows", [])) for t in curr_tables) >= 2:
                first_hdr = curr_tables[0].get("headers", [])
                # If Docling's table header is a genuine header (e.g. Name, Age, Year, Leases), respect Docling!
                if self.is_genuine_table_header(first_hdr) or self.is_table_header_row(first_hdr):
                    return None
                if not self.is_valid_continuation_data_row(first_hdr, num_cols):
                    return None
                # Special Case: Docling parsed rows, but its "header" is actually a cut-off DATA ROW!
                # (e.g. Best Buy p17 where 'Virginia 34 10 -' was falsely treated by Docling as the table header)
                header_line = "| " + " | ".join(headers) + " |"
                sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
                row1 = "| " + " | ".join(first_hdr) + " |"
                other_rows = ["| " + " | ".join(r) + " |" for r in curr_tables[0].get("rows", [])]
                table_md = header_line + "\n" + sep_line + "\n" + row1 + "\n" + "\n".join(other_rows)
                curr_lines = current_markdown.splitlines()
                start = max(0, curr_tables[0].get("start_line", 1) - 1)
                end_idx = start + 2 + len(curr_tables[0].get("rows", []))
                remaining_md = "\n".join(curr_lines[end_idx:]).strip()
                return table_md, remaining_md

        # Guard 3 & 4: Extract top table rows matching num_cols and verified as data rows
        rows, consumed_blocks = self.extract_top_table_rows(page, max_y=120.0, num_cols=num_cols)
        if not rows:
            return None

        # Build clean Markdown table with all captured rows
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
        data_lines = ["| " + " | ".join(r) + " |" for r in rows]
        table_md = header_line + "\n" + sep_line + "\n" + "\n".join(data_lines)

        # Remaining pieces below the table
        try:
            td = page.get_text("dict")
            blocks = [b for b in td.get("blocks", []) if b.get("type") == 0]
        except Exception:
            blocks = []

        remaining_pieces = []
        for b in blocks[consumed_blocks:]:
            b_text = "\n".join("".join(s.get("text", "") for s in l.get("spans", [])).strip() for l in b.get("lines", []))
            b_text = b_text.strip()
            if re.match(r"^\d{1,4}$", b_text) and b.get("bbox", [0, 0, 0, 0])[1] > 600:
                continue
            cleaned_piece = self.clean_pymupdf_text(b_text)
            if cleaned_piece and cleaned_piece != "_____________________________":
                remaining_pieces.append(cleaned_piece)

        remaining_md = "\n\n".join(remaining_pieces)
        return table_md, remaining_md

    def find_truly_missing_blocks(
        self,
        page: Any,
        markdown: str,
        prev_markdown: Optional[str] = None,
        next_markdown: Optional[str] = None,
    ) -> List[Tuple[float, str]]:
        """Find PyMuPDF text sub-blocks that are absent across [N-1, N, N+1] pages."""
        h = getattr(page.rect, "height", 792.0)
        curr_norm = self._normalize(markdown)
        prev_norm = self._normalize(prev_markdown or "")
        next_norm = self._normalize(next_markdown or "")

        blocks = page.get_text("blocks")
        recovered: List[Tuple[float, str]] = []

        for b in blocks:
            # Block type 0 is text
            if len(b) > 6 and b[6] != 0:
                continue

            y_top = b[1]
            y_bottom = b[3]

            # Skip header/footer bands if they are standard running headers or page numbers
            if y_top < self.margin_band or y_bottom > h - self.margin_band:
                lower_b = b[4].strip().lower()
                if re.match(r"^\d{1,4}$", lower_b) or "table of contents" in lower_b:
                    continue

            raw_block_text = b[4].strip()
            if len(raw_block_text) < self.min_char_len:
                continue

            # Sub-block splitting: evaluate each paragraph independently
            sub_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_block_text) if p.strip()]

            for sub_p in sub_paragraphs:
                if len(sub_p) < self.min_char_len:
                    continue

                # 1. Check current page
                if self.check_presence(sub_p, curr_norm) == "present":
                    continue

                # 2. Check previous page (cross-page merged text)
                if prev_norm and self.check_presence(sub_p, prev_norm) == "present":
                    continue

                # 3. Check next page (cross-page merged text)
                if next_norm and self.check_presence(sub_p, next_norm) == "present":
                    continue

                cleaned_text = self.clean_pymupdf_text(sub_p)
                recovered.append((y_top, cleaned_text))

        return recovered

    def find_missing_blocks(
        self,
        page: Any,
        markdown: str,
        prev_markdown: Optional[str] = None,
        next_markdown: Optional[str] = None,
    ) -> List[str]:
        """Convenience method returning just the text of truly missing blocks (for backwards compatibility)."""
        return [text for _, text in self.find_truly_missing_blocks(page, markdown, prev_markdown, next_markdown)]

    def recover(
        self,
        page: Any,
        markdown: str,
        raw_text: str,
        char_count: int,
        prev_markdown: Optional[str] = None,
        next_markdown: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Analyze page and apply optimal recovery if text or table cells were dropped.

        Returns:
            Tuple of (recovered_markdown, action_taken)
            action_taken can be: 'untouched', 'reconstructed_table_continuation',
                                 'reconstructed_false_table', or 'appended_orphan_blocks'
        """
        # Step 0: Check for Cross-Page Table Continuation (trailing row overflow from prev page table)
        continuation_result = self.try_reconstruct_table_continuation(page, prev_markdown, markdown)
        if continuation_result:
            table_md, remaining_md = continuation_result
            final_md = table_md + ("\n\n" + remaining_md if remaining_md else "")
            return final_md, "reconstructed_table_continuation"

        tables = extract_markdown_tables(markdown)

        # Step 1: Check for degenerate/collapsed table (false table from Docling)
        if self.is_degenerate_table(tables, markdown, raw_text, char_count):
            blocks = page.get_text("blocks")
            text_pieces: List[str] = []
            for b in blocks:
                if len(b) > 6 and b[6] == 0:
                    txt = self.clean_pymupdf_text(b[4].strip())
                    if txt:
                        text_pieces.append(txt)
            reconstructed = "\n\n".join(text_pieces)
            return reconstructed, "reconstructed_false_table"

        # Step 2: For genuine tables or mixed pages, rescue truly missing orphan blocks
        missing_tuples = self.find_truly_missing_blocks(page, markdown, prev_markdown, next_markdown)
        if not missing_tuples:
            return markdown, "untouched"

        h = getattr(page.rect, "height", 792.0)
        top_pieces = [text for y, text in missing_tuples if y < h * 0.35]
        bottom_pieces = [text for y, text in missing_tuples if y >= h * 0.35]

        result_md = markdown
        if top_pieces:
            result_md = "\n\n".join(top_pieces) + "\n\n" + result_md
        if bottom_pieces:
            supplementary = (
                "\n\n### Ghi chú & Nội dung bổ sung (Recovered Unassigned Content)\n\n"
                + "\n\n".join(bottom_pieces)
            )
            result_md = result_md + supplementary

        return result_md, "appended_orphan_blocks"
