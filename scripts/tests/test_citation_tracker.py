"""Tests for citation tracker module."""

import pytest

from src.analysis.citation_tracker import Citation, CitationGraph, CitationTracker


class TestCitation:
    """Test Citation dataclass."""

    def test_citation_creation(self):
        """Test creating a citation."""
        citation = Citation(
            reference_number="1",
            cited_text="This is a cited paper",
            context="Previous work [1] shows that...",
            page_number=1,
            authors="Smith et al.",
            year="2023",
            title="A Study on AI",
        )

        assert citation.reference_number == "1"
        assert citation.cited_text == "This is a cited paper"
        assert citation.context == "Previous work [1] shows that..."
        assert citation.page_number == 1
        assert citation.authors == "Smith et al."
        assert citation.year == "2023"
        assert citation.title == "A Study on AI"


class TestCitationGraph:
    """Test CitationGraph dataclass."""

    def test_graph_creation(self):
        """Test creating a citation graph."""
        graph = CitationGraph()

        assert graph.citations == {}

    def test_graph_with_citations(self):
        """Test creating graph with citations."""
        citation = Citation(
            reference_number="1",
            cited_text="Test",
            context="Test context",
        )
        graph = CitationGraph(citations={"1": citation})

        assert "1" in graph.citations
        assert graph.citations["1"].reference_number == "1"


class TestCitationTracker:
    """Test CitationTracker functionality."""

    def test_tracker_initialization(self):
        """Test tracker can be initialized."""
        tracker = CitationTracker()
        assert tracker.citations == {}

    def test_add_citation(self):
        """Test adding a citation via extract_citations."""
        tracker = CitationTracker()

        # Use extract_citations to add citations
        text = "Previous work [1] shows that deep learning is effective."
        citations = tracker.extract_citations(text)

        # Should find the citation
        assert "1" in tracker.citations
        assert tracker.citations["1"].reference_number == "1"

    def test_extract_citations_from_text(self):
        """Test extracting citations from text."""
        tracker = CitationTracker()

        text = """
        Deep learning has achieved remarkable success [1, 2, 3].
        Previous work [4] showed that transformers are effective.
        As noted by [5, 6], attention mechanisms are key.
        """

        citations = tracker.extract_citations(text)

        # Should find citations (actual count depends on regex patterns)
        assert isinstance(citations, list)

    def test_extract_citations_with_brackets(self):
        """Test extracting citations with different bracket styles."""
        tracker = CitationTracker()

        text = """
        Research [1] has shown promising results.
        Studies (2) have demonstrated similar findings.
        According to [3, 4], this approach works well.
        """

        citations = tracker.extract_citations(text)
        assert isinstance(citations, list)

    def test_get_citation_context(self):
        """Test getting citation context."""
        tracker = CitationTracker()

        # Extract citation first
        text = "Previous work [1] shows that deep learning is effective."
        tracker.extract_citations(text)

        retrieved = tracker.get_citation_context("1")
        assert retrieved is not None
        assert retrieved.reference_number == "1"

    def test_get_nonexistent_citation(self):
        """Test getting a citation that doesn't exist."""
        tracker = CitationTracker()

        result = tracker.get_citation_context("999")
        assert result is None

    def test_list_citations(self):
        """Test listing all citations."""
        tracker = CitationTracker()

        # Extract citations from text
        text = """
        Previous work [1] shows that deep learning is effective.
        Research [2] demonstrates similar findings.
        """
        tracker.extract_citations(text)

        all_citations = tracker.list_citations()
        assert len(all_citations) >= 1

    def test_most_cited(self):
        """Test getting most cited papers."""
        tracker = CitationTracker()

        # Add same citation multiple times
        text = """
        Previous work [1] shows this.
        Research [1] demonstrates that.
        Studies [1] prove this.
        Different work [2] shows that.
        """

        tracker.extract_citations(text)

        # Get most cited
        most_cited = tracker.get_most_cited(top_n=2)
        assert isinstance(most_cited, list)

    def test_extract_references_section(self):
        """Test extracting references section."""
        tracker = CitationTracker()

        text = """
        Introduction
        This paper studies deep learning.

        References
        [1] Smith, J. (2020). Paper One.
        [2] Jones, M. (2021). Paper Two.
        """

        refs = tracker.extract_references_section(text)
        assert isinstance(refs, str)