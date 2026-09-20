"""Caption extractor for academic papers."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Caption:
    """Represents a figure or table caption."""

    figure_id: str
    figure_type: str  # "figure" or "table"
    caption: str
    page_number: int
    context_before: str = ""  # Text before the caption
    context_after: str = ""  # Text after the caption


class CaptionExtractor:
    """Extract captions for figures and tables."""

    # Figure caption patterns
    FIGURE_CAPTION_PATTERNS = [
        # "Figure 1: This is a caption"
        r"(?:Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)\s*[:\.\-]\s*(.+?)(?:\.|$)",
        # "Figure 1. This is a caption"
        r"(?:Figure|Fig\.?|图)\s*(\d+[a-zA-Z]?)\.\s*(.+?)(?:\.|$)",
    ]

    # Table caption patterns
    TABLE_CAPTION_PATTERNS = [
        r"(?:Table|表|Tab\.?)\s*(\d+[a-zA-Z]?)\s*[:\.\-]\s*(.+?)(?:\.|$)",
        r"(?:Table|表|Tab\.?)\s*(\d+[a-zA-Z]?)\.\s*(.+?)(?:\.|$)",
    ]

    def __init__(self):
        self._figure_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.FIGURE_CAPTION_PATTERNS
        ]
        self._table_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.TABLE_CAPTION_PATTERNS
        ]

    def extract_captions(self, text: str, page_number: int = 0) -> list[Caption]:
        """Extract all captions from text."""
        captions = []

        # Extract figure captions
        for pattern in self._figure_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                figure_id = match.group(1)
                caption_text = match.group(2).strip()

                # Get context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)

                caption = Caption(
                    figure_id=figure_id,
                    figure_type="figure",
                    caption=caption_text,
                    page_number=page_number,
                    context_before=text[start:match.start()],
                    context_after=text[match.end():end],
                )
                captions.append(caption)

        # Extract table captions
        for pattern in self._table_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                table_id = match.group(1)
                caption_text = match.group(2).strip()

                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)

                caption = Caption(
                    figure_id=table_id,
                    figure_type="table",
                    caption=caption_text,
                    page_number=page_number,
                    context_before=text[start:match.start()],
                    context_after=text[match.end():end],
                )
                captions.append(caption)

        return captions

    def extract_from_pages(self, pages: list, page_start: int = 1) -> list[Caption]:
        """Extract captions from multiple pages."""
        all_captions = []

        for i, page in enumerate(pages):
            page_num = page_start + i
            captions = self.extract_captions(page.text, page_num)
            all_captions.extend(captions)

        return all_captions

    def get_caption_by_id(self, captions: list[Caption], figure_id: str) -> Optional[Caption]:
        """Get a specific caption by ID."""
        for caption in captions:
            if caption.figure_id == figure_id:
                return caption
        return None
