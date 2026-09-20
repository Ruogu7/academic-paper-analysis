"""Figure detector for academic papers."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Figure:
    """Represents a figure in the paper."""

    figure_id: str
    figure_type: str  # "figure" or "table"
    caption: str = ""
    page_number: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None  # x0, y0, x1, y1
    image_path: Optional[str] = None


class FigureDetector:
    """Detect figures and tables in academic papers."""

    # Figure patterns
    FIGURE_PATTERNS = [
        r"(?:Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)",
        r"图\s*(\d+[a-zA-Z]?)",
    ]

    # Table patterns
    TABLE_PATTERNS = [
        r"(?:Table|表|Tab\.?)\s*(\d+[a-zA-Z]?)",
    ]

    # Caption patterns
    CAPTION_PATTERNS = [
        r"(?:Figure|Fig\.?|图)\s*\d+[a-zA-Z]?\s*[:\.\-]?\s*(.+?)(?:\.|$)",
        r"(?:Table|表|Tab\.?)\s*\d+[a-zA-Z]?\s*[:\.\-]?\s*(.+?)(?:\.|$)",
    ]

    def __init__(self):
        self._figure_patterns = [re.compile(p, re.IGNORECASE) for p in self.FIGURE_PATTERNS]
        self._table_patterns = [re.compile(p, re.IGNORECASE) for p in self.TABLE_PATTERNS]
        self._caption_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.CAPTION_PATTERNS]

    def detect_figures(self, text: str, page_number: int = 0) -> list[Figure]:
        """Detect figures in the text."""
        figures = []

        for pattern in self._figure_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                figure_id = match.group(1)
                caption = self._extract_caption(text, match.end())

                figure = Figure(
                    figure_id=figure_id,
                    figure_type="figure",
                    caption=caption,
                    page_number=page_number,
                )
                figures.append(figure)

        return figures

    def detect_tables(self, text: str, page_number: int = 0) -> list[Figure]:
        """Detect tables in the text."""
        tables = []

        for pattern in self._table_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                table_id = match.group(1)
                caption = self._extract_caption(text, match.end())

                table = Figure(
                    figure_id=table_id,
                    figure_type="table",
                    caption=caption,
                    page_number=page_number,
                )
                tables.append(table)

        return tables

    def detect_all(self, text: str, page_number: int = 0) -> list[Figure]:
        """Detect all figures and tables."""
        figures = self.detect_figures(text, page_number)
        tables = self.detect_tables(text, page_number)
        return figures + tables

    def _extract_caption(self, text: str, end_pos: int) -> str:
        """Extract caption following a figure/table reference."""
        # Get text after the reference
        after_text = text[end_pos:end_pos + 200]

        for pattern in self._caption_patterns:
            match = pattern.search(after_text)
            if match:
                caption = match.group(1).strip()
                # Clean up caption
                caption = re.sub(r"^\s*[:\.\-]\s*", "", caption)
                caption = caption[:200]  # Limit length
                return caption

        return ""

    def get_figure_by_id(self, figures: list[Figure], figure_id: str) -> Optional[Figure]:
        """Get a specific figure by ID."""
        for figure in figures:
            if figure.figure_id == figure_id:
                return figure
        return None
