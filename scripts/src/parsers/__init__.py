"""PDF and document parsers."""

from .pdf_parser import PDFParser
from .html_parser import HTMLParser, UniversalParser, ParsedPage
from .section_detector import SectionDetector, Section, SectionType
from .figure_detector import FigureDetector, Figure
from .figure_extractor import FigureExtractor, ExtractedFigure
from .caption_extractor import CaptionExtractor, Caption

__all__ = [
    "PDFParser",
    "HTMLParser",
    "UniversalParser",
    "ParsedPage",
    "SectionDetector",
    "Section",
    "SectionType",
    "FigureDetector",
    "Figure",
    "FigureExtractor",
    "ExtractedFigure",
    "CaptionExtractor",
    "Caption",
]
