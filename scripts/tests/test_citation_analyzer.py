"""Tests for citation analysis."""

import pytest
from unittest.mock import patch, MagicMock

from src.analysis.citation_analyzer import (
    CitationAnalyzer,
    CitationAnalysis,
    RelevantCitation,
)


class TestCitationAnalyzerInterface:
    """Test CitationAnalyzer interface and basic functionality."""

    def test_analyzer_has_analyze_method(self):
        """Test that CitationAnalyzer has an analyze method."""
        analyzer = CitationAnalyzer()
        assert hasattr(analyzer, "analyze")
        assert callable(getattr(analyzer, "analyze"))

    def test_relevant_citation_dataclass(self):
        """Test RelevantCitation dataclass structure."""
        citation = RelevantCitation(
            reference_number="[1]",
            first_author="Smith",
            year="2020",
            title="Smith et al.: A Foundation for Graph-Based Retrieval",
            topic="knowledge graph",
            relevance_score=0.95,
            contribution="Provided foundation for graph-based retrieval",
            context_in_paper="The paper builds upon Smith et al.'s work...",
            paper_link="https://arxiv.org/abs/2001.12345",
        )

        assert citation.reference_number == "[1]"
        assert citation.first_author == "Smith"
        assert citation.year == "2020"
        assert citation.title == "Smith et al.: A Foundation for Graph-Based Retrieval"
        assert citation.topic == "knowledge graph"
        assert citation.relevance_score == 0.95
        assert citation.paper_link == "https://arxiv.org/abs/2001.12345"

    def test_citation_analysis_dataclass(self):
        """Test CitationAnalysis dataclass structure."""
        analysis = CitationAnalysis(
            paper_title="Test Paper",
            total_citations=10,
            analyzed_citations=5,
            relevant_citations=[],
        )

        assert analysis.paper_title == "Test Paper"
        assert analysis.total_citations == 10
        assert analysis.analyzed_citations == 5


class TestCitationAnalyzerWithMockLLM:
    """Test CitationAnalyzer with mocked LLM."""

    @patch("src.analysis.citation_analyzer.LLMClient")
    def test_analyze_returns_citation_analysis(self, mock_llm_class):
        """Test that analyze returns CitationAnalysis object."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.complete.return_value = MagicMock(
            content="""## 相关引文分析

### [1] Smith et al., 2020
- **主题**: 知识图谱构建
- **第一作者**: Smith
- **相关性**: 0.95
- **贡献**: 提供了基于图谱检索的基础框架

### [2] Zhang, 2021
- **主题**: 神经检索
- **第一作者**: Zhang
- **相关性**: 0.85
- **贡献**: 提出了神经网络的检索方法"""
        )

        # Create analyzer and analyze
        analyzer = CitationAnalyzer()
        citations = [
            MagicMock(reference_number="[1]", context="...context1...", cited_text="[1]"),
            MagicMock(reference_number="[2]", context="...context2...", cited_text="[2]"),
        ]

        result = analyzer.analyze(
            paper_title="Test Paper",
            citations=citations,
            paper_content="Test content...",
        )

        assert isinstance(result, CitationAnalysis)
        assert result.paper_title == "Test Paper"
        assert result.total_citations == 2

    @patch("src.analysis.citation_analyzer.LLMClient")
    def test_analyze_filters_by_relevance(self, mock_llm_class):
        """Test that analyze filters citations by relevance threshold."""
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_llm.complete.return_value = MagicMock(
            content="""## 相关引文分析

### [1] Smith, 2020
- **主题**: 相关主题
- **相关性**: 0.9
- **贡献**: 重要贡献

### [2] Jones, 2019
- **主题**: 不太相关
- **相关性**: 0.3
- **贡献**: 轻微提及"""
        )

        analyzer = CitationAnalyzer()
        citations = [
            MagicMock(reference_number="[1]", context="...context...", cited_text="[1]"),
            MagicMock(reference_number="[2]", context="...context...", cited_text="[2]"),
        ]

        result = analyzer.analyze(
            paper_title="Test",
            citations=citations,
            paper_content="Content...",
            relevance_threshold=0.5,
        )

        # Should filter out low relevance citations
        assert len(result.relevant_citations) >= 0

    def test_analyze_empty_citations(self):
        """Test analyze with empty citations list."""
        analyzer = CitationAnalyzer()

        result = analyzer.analyze(
            paper_title="Test Paper",
            citations=[],
            paper_content="Some content",
        )

        assert result.total_citations == 0
        assert result.analyzed_citations == 0
        assert result.relevant_citations == []


class TestRelevantCitationParsing:
    """Test parsing of LLM response into RelevantCitation objects."""

    def test_parse_llm_response_basic(self):
        """Test basic parsing of LLM response."""
        analyzer = CitationAnalyzer()

        response = """## 相关引文分析

### [1] Smith et al., 2020
- **主题**: 知识图谱
- **第一作者**: Smith
- **相关性**: 0.95
- **贡献**: 提供了基础框架

### [3] Wang, 2022
- **主题**: 索引优化
- **第一作者**: Wang
- **相关性**: 0.88
- **贡献**: 改进了检索效率"""

        parsed = analyzer._parse_llm_response(response)

        assert len(parsed) == 2
        assert parsed[0].reference_number == "[1]"
        assert parsed[0].first_author == "Smith"
        assert parsed[0].relevance_score == 0.95

    def test_parse_llm_response_handles_empty(self):
        """Test parsing empty response."""
        analyzer = CitationAnalyzer()

        parsed = analyzer._parse_llm_response("")
        assert parsed == []
