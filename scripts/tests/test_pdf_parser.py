"""Tests for PDF parser module."""

import pytest
from pathlib import Path

from src.parsers.pdf_parser import PDFParser, PDFMetadata, PDFPage, PDFDocument


def find_test_pdf() -> Path | None:
    """Find a PDF file in common locations for testing."""
    search_paths = [
        Path("."),
        Path(".."),
        Path("~/Downloads"),
    ]
    extensions = [".pdf"]

    for base in search_paths:
        if not base.exists():
            continue
        for ext in extensions:
            # Look for common paper filenames
            for name in ["paper", "test", "sample"]:
                for p in base.rglob(f"*{name}*{ext}"):
                    if p.is_file():
                        return p
    return None


@pytest.fixture
def pdf_path() -> Path:
    """Fixture providing path to a test PDF.

    Searches current directory and parent directories for PDF files.
    Skips tests if no PDF found.
    """
    pdf = find_test_pdf()
    if pdf is None:
        pytest.skip("No PDF file found for testing")
    return pdf


class TestPDFParser:
    """Test PDFParser functionality."""

    def test_parser_initialization(self, pdf_path: Path):
        """Test parser can be initialized with PDF path."""
        parser = PDFParser(pdf_path)
        assert parser.path == pdf_path

    def test_parse_pdf(self, pdf_path: Path):
        """Test parsing a PDF file."""
        parser = PDFParser(pdf_path)
        doc = parser.parse()

        assert doc is not None
        assert len(doc.pages) > 0

    def test_extract_metadata(self, pdf_path: Path):
        """Test metadata extraction from PDF."""
        parser = PDFParser(pdf_path)
        doc = parser.parse()

        # Should have some metadata (even if default values)
        assert doc.metadata is not None

    def test_get_full_text(self, pdf_path: Path):
        """Test extracting full text from PDF."""
        parser = PDFParser(pdf_path)
        _ = parser.parse()

        full_text = parser.get_full_text()

        assert full_text is not None
        assert len(full_text) > 0
        assert isinstance(full_text, str)

    def test_page_count(self, pdf_path: Path):
        """Test correct page count."""
        parser = PDFParser(pdf_path)
        doc = parser.parse()

        # RAG paper should have around 19 pages
        assert len(doc.pages) >= 10
        assert len(doc.pages) <= 30

    def test_get_page_from_document(self, pdf_path: Path):
        """Test getting page from parsed document."""
        parser = PDFParser(pdf_path)
        doc = parser.parse()

        first_page = doc.pages[0]
        assert first_page is not None
        assert first_page.page_number == 1
        assert len(first_page.text) > 0


class TestPDFMetadata:
    """Test PDFMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata with required fields."""
        metadata = PDFMetadata(
            title="Test Paper",
            author="John Doe",
            subject="AI",
            creator="Test Creator",
            producer="Test Producer",
        )

        assert metadata.title == "Test Paper"
        assert metadata.author == "John Doe"
        assert metadata.subject == "AI"
        assert metadata.creator == "Test Creator"
        assert metadata.producer == "Test Producer"

    def test_metadata_with_optional_fields(self):
        """Test creating metadata with optional fields."""
        metadata = PDFMetadata(
            title="Test Paper",
            author="John Doe",
            subject="AI",
            creator="Test Creator",
            producer="Test Producer",
            creation_date="2024-01-01",
            modification_date="2024-01-02",
        )

        assert metadata.creation_date == "2024-01-01"
        assert metadata.modification_date == "2024-01-02"


class TestPDFPage:
    """Test PDFPage dataclass."""

    def test_page_creation(self):
        """Test creating a PDF page."""
        page = PDFPage(
            page_number=1,
            text="Test content",
            raw_text="Raw test content",
        )

        assert page.page_number == 1
        assert page.text == "Test content"
        assert page.raw_text == "Raw test content"


class TestPDFDocument:
    """Test PDFDocument dataclass."""

    def test_document_creation(self):
        """Test creating a PDF document."""
        doc = PDFDocument(
            path=Path("test.pdf"),
            metadata=None,
            pages=[],
            total_pages=0,
        )

        assert doc.path == Path("test.pdf")
        assert doc.metadata is None
        assert len(doc.pages) == 0
        assert doc.total_pages == 0
