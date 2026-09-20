"""Tests for HTML parser module."""

import pytest
from pathlib import Path

from src.parsers.html_parser import HTMLParser, UniversalParser, ParsedPage


class TestParsedPage:
    """Test ParsedPage dataclass."""

    def test_parsed_page_creation(self):
        """Test creating a ParsedPage."""
        page = ParsedPage(
            title="Test Title",
            text="Test content",
            sections={"Introduction": "Intro text"},
            metadata={"author": "John Doe"},
            links=[{"text": "Link", "href": "http://example.com"}],
            figures=[{"src": "img.png", "alt": "Image", "caption": "Caption"}],
        )

        assert page.title == "Test Title"
        assert page.text == "Test content"
        assert page.sections == {"Introduction": "Intro text"}
        assert page.metadata == {"author": "John Doe"}
        assert len(page.links) == 1
        assert len(page.figures) == 1


class TestHTMLParser:
    """Test HTMLParser functionality."""

    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = HTMLParser()
        assert parser is not None

    def test_parse_simple_html(self):
        """Test parsing simple HTML content."""
        parser = HTMLParser()

        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Paper</title></head>
        <body>
            <h1>Introduction</h1>
            <p>This is a test paper about machine learning.</p>
        </body>
        </html>
        """

        result = parser.parse_html(html)

        assert result.title == "Test Paper"
        assert "machine learning" in result.text.lower()

    def test_extract_title_from_h1(self):
        """Test extracting title from h1 tag."""
        parser = HTMLParser()

        html = """
        <html>
        <body>
            <h1>My Research Paper</h1>
        </body>
        </html>
        """

        result = parser.parse_html(html)
        assert result.title == "My Research Paper"

    def test_extract_metadata(self):
        """Test extracting metadata from HTML."""
        parser = HTMLParser()

        html = """
        <html>
        <head>
            <meta name="author" content="John Doe">
            <meta name="description" content="A paper about AI">
            <meta name="keywords" content="AI, Machine Learning">
        </head>
        <body></body>
        </html>
        """

        result = parser.parse_html(html)

        assert result.metadata.get("author") == "John Doe"
        assert result.metadata.get("description") == "A paper about AI"
        assert "AI" in result.metadata.get("keywords", "")

    def test_extract_sections_from_headings(self):
        """Test extracting sections from heading tags."""
        parser = HTMLParser()

        html = """
        <html>
        <body>
            <h1>Main Title</h1>
            <h2>Introduction</h2>
            <p>Intro text here.</p>
            <h2>Methods</h2>
            <p>Methods text here.</p>
            <h2>Results</h2>
            <p>Results text here.</p>
        </body>
        </html>
        """

        result = parser.parse_html(html)

        assert "Main Title" in result.sections or len(result.sections) > 0

    def test_extract_links(self):
        """Test extracting links from HTML."""
        parser = HTMLParser()

        html = """
        <html>
        <body>
            <a href="http://example.com">Example</a>
            <a href="https://paper.org/paper.pdf">Paper</a>
        </body>
        </html>
        """

        result = parser.parse_html(html)

        assert len(result.links) >= 1

    def test_extract_figures(self):
        """Test extracting images from HTML."""
        parser = HTMLParser()

        html = """
        <html>
        <body>
            <img src="figure1.png" alt="Figure 1">
            <img src="figure2.jpg" alt="Figure 2">
        </body>
        </html>
        """

        result = parser.parse_html(html)

        assert len(result.figures) >= 1


class TestUniversalParser:
    """Test UniversalParser functionality."""

    def test_universal_parser_initialization(self):
        """Test universal parser can be initialized."""
        parser = UniversalParser()
        assert parser is not None
        assert parser.html_parser is not None

    def test_parse_html_file(self, tmp_path):
        """Test parsing HTML file."""
        # Create a temporary HTML file
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body><p>Content</p></body>
        </html>
        """)

        parser = UniversalParser()
        result = parser.parse(html_file)

        assert result.title == "Test"
        assert "Content" in result.text

    def test_unsupported_format(self, tmp_path):
        """Test handling unsupported file format."""
        parser = UniversalParser()

        # Create a file with unsupported extension
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some text")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(test_file)
