"""Tests for citation graph module."""

import pytest

from src.analysis.citation_graph import (
    CitationGraphVisualizer,
    CitationGraphData,
    CitationNode,
    CitationEdge,
)


class TestCitationNode:
    """Test CitationNode dataclass."""

    def test_node_creation(self):
        """Test creating a citation node."""
        node = CitationNode(
            id="1",
            label="Smith et al. 2020",
            year=2020,
            authors="Smith, John",
            title="A Study on AI",
            node_type="paper",
            cited_count=10,
        )

        assert node.id == "1"
        assert node.label == "Smith et al. 2020"
        assert node.year == 2020
        assert node.authors == "Smith, John"
        assert node.title == "A Study on AI"
        assert node.node_type == "paper"
        assert node.cited_count == 10

    def test_node_with_defaults(self):
        """Test creating node with default values."""
        node = CitationNode(id="1", label="Test")

        assert node.id == "1"
        assert node.label == "Test"
        assert node.year is None
        assert node.authors == ""
        assert node.node_type == "paper"


class TestCitationEdge:
    """Test CitationEdge dataclass."""

    def test_edge_creation(self):
        """Test creating a citation edge."""
        edge = CitationEdge(source="root", target="1", edge_type="cites")

        assert edge.source == "root"
        assert edge.target == "1"
        assert edge.edge_type == "cites"

    def test_edge_with_defaults(self):
        """Test creating edge with default values."""
        edge = CitationEdge(source="1", target="2")

        assert edge.source == "1"
        assert edge.target == "2"
        assert edge.edge_type == "cites"


class TestCitationGraphData:
    """Test CitationGraphData dataclass."""

    def test_graph_data_creation(self):
        """Test creating graph data."""
        nodes = [
            CitationNode(id="root", label="Main Paper"),
            CitationNode(id="1", label="Citation 1"),
        ]
        edges = [CitationEdge(source="root", target="1")]

        graph = CitationGraphData(
            nodes=nodes,
            edges=edges,
            root_paper="Main Paper",
        )

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.root_paper == "Main Paper"

    def test_graph_data_with_defaults(self):
        """Test creating graph data with defaults."""
        graph = CitationGraphData()

        assert graph.nodes == []
        assert graph.edges == []
        assert graph.root_paper == ""


class TestCitationGraphVisualizer:
    """Test CitationGraphVisualizer functionality."""

    def test_visualizer_initialization(self):
        """Test visualizer can be initialized."""
        viz = CitationGraphVisualizer()
        assert viz is not None

    def test_generate_mermaid(self):
        """Test generating Mermaid diagram."""
        viz = CitationGraphVisualizer()

        graph = CitationGraphData(
            nodes=[
                CitationNode(id="root", label="Main Paper"),
                CitationNode(id="1", label="Citation 1", year=2020),
            ],
            edges=[CitationEdge(source="root", target="1")],
            root_paper="Main Paper",
        )

        mermaid = viz.generate_mermaid(graph)

        assert "flowchart" in mermaid
        assert "root" in mermaid
        assert "Main Paper" in mermaid

    def test_generate_mermaid_with_different_directions(self):
        """Test generating Mermaid with different directions."""
        viz = CitationGraphVisualizer()

        graph = CitationGraphData(
            nodes=[CitationNode(id="root", label="Main")],
            edges=[],
        )

        # Test LR direction
        mermaid_lr = viz.generate_mermaid(graph, direction="LR")
        assert "flowchart LR" in mermaid_lr

    def test_generate_mermaid_with_edge_types(self):
        """Test Mermaid generation with different edge types."""
        viz = CitationGraphVisualizer()

        graph = CitationGraphData(
            nodes=[
                CitationNode(id="root", label="Main"),
                CitationNode(id="1", label="Method", node_type="method"),
                CitationNode(id="2", label="Dataset", node_type="dataset"),
            ],
            edges=[
                CitationEdge(source="root", target="1", edge_type="uses"),
                CitationEdge(source="root", target="2", edge_type="extends"),
            ],
        )

        mermaid = viz.generate_mermaid(graph)
        assert "flowchart" in mermaid

    def test_build_graph_from_tracker(self):
        """Test building graph from citation tracker."""
        from src.analysis.citation_tracker import CitationTracker

        viz = CitationGraphVisualizer()

        # Extract citations from text
        tracker = CitationTracker()
        text = "Previous work [1] shows that deep learning is effective."
        tracker.extract_citations(text)

        # Build graph
        graph = viz.build_graph_from_tracker(tracker, "Test Paper")

        # Should have root node and at least one citation
        assert len(graph.nodes) >= 1
        assert graph.root_paper == "Test Paper"

    def test_save_graph_data(self, tmp_path):
        """Test saving graph data to JSON."""
        import json

        viz = CitationGraphVisualizer()

        graph = CitationGraphData(
            nodes=[CitationNode(id="1", label="Test")],
            edges=[],
            root_paper="Test",
        )

        output_file = tmp_path / "graph.json"
        viz.save_graph_data(graph, output_file)

        # Check file was created
        assert output_file.exists()

        # Check content
        data = json.loads(output_file.read_text())
        assert "nodes" in data
        assert "edges" in data

    def test_generate_html(self, tmp_path):
        """Test generating HTML visualization."""
        viz = CitationGraphVisualizer()

        graph = CitationGraphData(
            nodes=[
                CitationNode(id="root", label="Main Paper"),
                CitationNode(id="1", label="Citation 1", year=2020),
            ],
            edges=[CitationEdge(source="root", target="1")],
            root_paper="Main Paper",
        )

        output_file = tmp_path / "graph.html"
        html = viz.generate_html(graph, output_file)

        # Should return HTML string
        assert isinstance(html, str)
        assert "<html" in html.lower()

        # Should save file
        assert output_file.exists()
