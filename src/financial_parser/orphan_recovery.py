"""Orphan text recovery for Docling TableFormer dropped cells and collapsed tables.

Follows the recovery architecture outlined in `idea/docling_tableformer_orphan_cells_fix.md`:
1. Identifies false/degenerate tables where TableFormer collapsed narrative text into a single
   broken row or header, and reconstructs clean structured text blocks.
2. Identifies unassigned orphan text blocks outside the predicted grid of genuine tables
   and attaches them under a structured supplementary content section.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from .markdown_utils import extract_markdown_tables


class OrphanTextRecoverer:
    """Recovers text and numbers dropped by Docling table structure recognition."""

    def __init__(
        self,
        min_char_len: int = 15,
        missing_ratio_threshold: float = 0.35,
        min_missing_words: int = 3,
        huge_cell_threshold: int = 500,
    ) -> None:
        self.min_char_len = min_char_len
        self.missing_ratio_threshold = missing_ratio_threshold
        self.min_missing_words = min_missing_words
        self.huge_cell_threshold = huge_cell_threshold

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

    def find_missing_blocks(
        self,
        page: Any,
        markdown: str,
    ) -> List[str]:
        """Find PyMuPDF text blocks that are absent from the parsed markdown."""
        h = getattr(page.rect, "height", 792.0)
        md_clean = re.sub(r"[^a-zA-Z0-9]+", " ", markdown.lower())
        md_words = set(md_clean.split())

        blocks = page.get_text("blocks")
        recovered: List[str] = []

        for b in blocks:
            # Block type 0 is text
            if len(b) > 6 and b[6] != 0:
                continue

            # Skip header/footer bands (first and last 35 points) if they are standard running headers
            if b[1] < 35 or b[3] > h - 35:
                lower_txt = b[4].strip().lower()
                if "table of contents" in lower_txt or re.match(r"^\d{1,4}$", lower_txt):
                    continue

            text = b[4].strip()
            if len(text) < self.min_char_len:
                continue

            words = re.findall(r"[a-zA-Z0-9]+", text.lower())
            if not words:
                continue

            missing = [w for w in words if w not in md_words]
            missing_ratio = len(missing) / len(words)
            text_norm = re.sub(r"\s+", " ", text.strip().lower())
            sample = text_norm[:min(30, len(text_norm))]

            if sample not in markdown.lower() and (
                missing_ratio >= self.missing_ratio_threshold
                or len(missing) >= self.min_missing_words
            ):
                recovered.append(text)

        return recovered

    def recover(
        self,
        page: Any,
        markdown: str,
        raw_text: str,
        char_count: int,
    ) -> Tuple[str, str]:
        """Analyze page and apply optimal recovery if text or table cells were dropped.

        Returns:
            Tuple of (recovered_markdown, action_taken)
            action_taken can be: 'untouched', 'reconstructed_false_table', or 'appended_orphan_blocks'
        """
        tables = extract_markdown_tables(markdown)

        # Step 1: Check for degenerate/collapsed table
        if self.is_degenerate_table(tables, markdown, raw_text, char_count):
            blocks = page.get_text("blocks")
            text_pieces: List[str] = []
            for b in blocks:
                if len(b) > 6 and b[6] == 0:
                    txt = b[4].strip()
                    if txt:
                        text_pieces.append(txt)
            reconstructed = "\n\n".join(text_pieces)
            return reconstructed, "reconstructed_false_table"

        # Step 2: For genuine tables or mixed pages, rescue orphan blocks
        missing_blocks = self.find_missing_blocks(page, markdown)
        if missing_blocks:
            supplementary_section = (
                "\n\n### Ghi chú & Nội dung bổ sung chưa gán bảng (Recovered Unassigned Content)\n\n"
                + "\n\n".join(missing_blocks)
            )
            return markdown + supplementary_section, "appended_orphan_blocks"

        return markdown, "untouched"
