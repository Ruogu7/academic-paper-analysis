"""HTML parser for academic papers."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


@dataclass
class ParsedPage:
    """Represents a parsed HTML page."""

    title: str
    text: str
    sections: dict[str, str]  # section_name -> content
    metadata: dict[str, str]
    links: list[dict]  # {"text": str, "href": str}
    figures: list[dict]  # {"src": str, "alt": str, "caption": str}


class HTMLParser:
    """Parse HTML format academic papers."""

    def __init__(self):
        if BeautifulSoup is None:
            raise ImportError("BeautifulSoup4 is required. Install with: pip install beautifulsoup4")

    def parse_file(self, html_path: str | Path) -> ParsedPage:
        """Parse an HTML file."""
        html_path = Path(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self.parse_html(html_content)

    def parse_html(self, html_content: str) -> ParsedPage:
        """Parse HTML content."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = self._extract_title(soup)

        # Extract metadata
        metadata = self._extract_metadata(soup)

        # Extract main text content
        text = self._extract_text(soup)

        # Extract sections
        sections = self._extract_sections(soup)

        # Extract links
        links = self._extract_links(soup)

        # Extract figures
        figures = self._extract_figures(soup)

        return ParsedPage(
            title=title,
            text=text,
            sections=sections,
            metadata=metadata,
            links=links,
            figures=figures,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML."""
        # Try <title> tag first
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        # Try <h1> tag
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        # Try meta tags
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"]

        return "Untitled"

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract metadata from HTML."""
        metadata = {}

        # Common meta tags
        meta_fields = {
            "author": ["author", "dc.creator"],
            "description": ["description", "dc.description"],
            "keywords": ["keywords", "dc.subject"],
            "doi": ["doi", "dc.identifier"],
            "published": ["published", "dc.date"],
            "journal": ["journal", "dc.publisher"],
        }

        for field, selectors in meta_fields.items():
            for selector in selectors:
                if selector.startswith("dc."):
                    meta_tag = soup.find("meta", attrs={"name": selector}) or soup.find(
                        "meta", attrs={"property": selector}
                    )
                else:
                    meta_tag = soup.find("meta", attrs={"name": selector, "content": True}) or soup.find(
                        "meta", attrs={"property": selector, "content": True}
                    )
                if meta_tag and meta_tag.get("content"):
                    metadata[field] = meta_tag["content"]
                    break

        return metadata

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content."""
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        # Try to find main content area
        main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|body", re.I))

        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _extract_sections(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract sections from HTML based on heading tags."""
        sections = {}
        current_section = "Introduction"
        current_content = []

        # Find all headings
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])
            text = heading.get_text(strip=True)

            if level == 1:
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = text
                current_content = []
            else:
                # Add to current section
                current_content.append(f"## {text}\n")

                # Get next siblings until next heading
                for sibling in heading.find_next_siblings():
                    if sibling.name and sibling.name.startswith("h"):
                        break
                    if sibling.name in ["p", "ul", "ol", "table", "div"]:
                        text = sibling.get_text(strip=True)
                        if text:
                            current_content.append(text)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # If no sections found, try common academic paper section names
        if not sections:
            text = self._extract_text(soup)
            section_patterns = {
                "Abstract": r"(?:abstract|摘要)[:\s]*(.{100,1000}?)(?:introduction|1\.|一、)",
                "Introduction": r"(?:introduction|引言|1\.)[:\s]*(.{100,3000}?)(?:method|2\.|二、)",
                "Methods": r"(?:method|方法|2\.)[:\s]*(.{100,3000}?)(?:result|3\.|三、)",
                "Results": r"(?:result|结果|3\.)[:\s]*(.{100,3000}?)(?:discussion|4\.|四、)",
                "Discussion": r"(?:discussion|讨论|4\.)[:\s]*(.{100,3000}?)(?:conclusion|5\.|五、)",
                "Conclusion": r"(?:conclusion|结论|5\.)[:\s]*(.{100,2000}?)(?:reference|bibliography|$)",
            }

            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    sections[section_name] = match.group(1).strip()

        return sections

    def _extract_links(self, soup: BeautifulSoup) -> list[dict]:
        """Extract links from HTML."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if href and (text or href.startswith("http")):
                links.append({"text": text or href, "href": href})
        return links

    def _extract_figures(self, soup: BeautifulSoup) -> list[dict]:
        """Extract figures from HTML."""
        figures = []

        # Find img tags
        for img in soup.find_all("img"):
            figure = {
                "src": img.get("src", ""),
                "alt": img.get("alt", ""),
                "caption": "",
            }

            # Try to find caption
            parent = img.parent
            if parent:
                # Check for figure/caption structure
                figure_parent = parent.find_parent("figure")
                if figure_parent:
                    caption_tag = figure_parent.find("figcaption")
                    if caption_tag:
                        figure["caption"] = caption_tag.get_text(strip=True)
                else:
                    # Check for nearby caption
                    next_caption = parent.find_next_sibling(["p", "div"], string=re.compile(r"fig\.?\s*\d", re.I))
                    if next_caption:
                        figure["caption"] = next_caption.get_text(strip=True)

            if figure["src"]:
                figures.append(figure)

        return figures


class UniversalParser:
    """Universal parser that handles both PDF and HTML."""

    def __init__(self):
        self.html_parser = HTMLParser()

    def parse(self, file_path: str | Path) -> ParsedPage:
        """Parse a document (PDF or HTML)."""
        from .pdf_parser import PDFParser

        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            # Create PDF parser with the file path
            pdf_parser = PDFParser(file_path)
            pdf_result = pdf_parser.parse()
            return self._convert_pdf_result(pdf_result)
        elif suffix in [".html", ".htm"]:
            return self.html_parser.parse_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _convert_pdf_result(self, pdf_result) -> ParsedPage:
        """Convert PDF parse result to unified format."""
        # Import here to avoid circular dependency
        from .section_detector import SectionDetector

        detector = SectionDetector()
        sections = detector.detect_sections(pdf_result.text)

        return ParsedPage(
            title=pdf_result.title,
            text=pdf_result.text,
            sections=sections,
            metadata=pdf_result.metadata,
            links=[],  # PDF doesn't have links
            figures=[],  # PDF figures handled separately
        )
