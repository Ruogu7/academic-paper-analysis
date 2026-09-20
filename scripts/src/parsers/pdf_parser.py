"""PDF Parser for academic papers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pypdf
from loguru import logger


@dataclass
class PDFMetadata:
    """Metadata extracted from PDF."""

    title: str
    author: str
    subject: str
    creator: str
    producer: str
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None


@dataclass
class PDFPage:
    """Single page content."""

    page_number: int
    text: str
    raw_text: str


@dataclass
class PDFDocument:
    """Complete PDF document representation."""

    path: Path
    metadata: Optional[PDFMetadata]
    pages: list[PDFPage]
    total_pages: int


class PDFParser:
    """Parser for extracting content from PDF files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader: Optional[pypdf.PdfReader] = None

    def _ensure_loaded(self) -> None:
        """Lazy load the PDF file."""
        if self._reader is None:
            logger.info(f"Loading PDF: {self.path}")
            self._reader = pypdf.PdfReader(str(self.path))

    @property
    def reader(self) -> pypdf.PdfReader:
        """Get the PDF reader instance."""
        self._ensure_loaded()
        return self._reader

    def parse(self) -> PDFDocument:
        """Parse the entire PDF document."""
        self._ensure_loaded()

        # Extract metadata
        metadata = self._extract_metadata()

        # Extract pages
        pages = []
        for i, page in enumerate(self.reader.pages):
            text = page.extract_text() or ""
            pages.append(
                PDFPage(
                    page_number=i + 1,
                    text=text.strip(),
                    raw_text=text,
                )
            )

        logger.info(f"Parsed {len(pages)} pages from {self.path.name}")

        return PDFDocument(
            path=self.path,
            metadata=metadata,
            pages=pages,
            total_pages=len(pages),
        )

    def _extract_metadata(self) -> Optional[PDFMetadata]:
        """Extract metadata from PDF."""
        try:
            info = self.reader.metadata
            if info is None:
                return None

            return PDFMetadata(
                title=info.get("/Title", "").strip() or "Unknown Title",
                author=info.get("/Author", "").strip() or "Unknown Author",
                subject=info.get("/Subject", "").strip() or "",
                creator=info.get("/Creator", "").strip() or "",
                producer=info.get("/Producer", "").strip() or "",
                creation_date=info.get("/CreationDate", ""),
                modification_date=info.get("/ModDate", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return None

    def get_full_text(self) -> str:
        """Get the full text content of the document."""
        doc = self.parse()
        return "\n\n".join(page.text for page in doc.pages)

    def get_text_by_page_range(self, start: int, end: int) -> str:
        """Get text from a specific page range (1-indexed)."""
        doc = self.parse()
        pages = doc.pages[start - 1 : end]
        return "\n\n".join(page.text for page in pages)
