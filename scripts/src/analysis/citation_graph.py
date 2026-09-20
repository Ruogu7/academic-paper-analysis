"""Citation graph visualization module."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class CitationNode:
    """Represents a node in the citation graph."""

    id: str
    label: str
    year: Optional[int] = None
    authors: str = ""
    title: str = ""
    node_type: str = "paper"  # "paper", "method", "dataset"
    cited_count: int = 0


@dataclass
class CitationEdge:
    """Represents an edge in the citation graph."""

    source: str
    target: str
    edge_type: str = "cites"  # "cites", "extends", "uses"


@dataclass
class CitationGraphData:
    """Represents the complete citation graph."""

    nodes: list[CitationNode] = field(default_factory=list)
    edges: list[CitationEdge] = field(default_factory=list)
    root_paper: str = ""


class CitationGraphVisualizer:
    """Visualize citation graphs in various formats."""

    def __init__(self):
        pass

    def generate_mermaid(
        self,
        graph_data: CitationGraphData,
        direction: str = "TD",
    ) -> str:
        """Generate Mermaid diagram code."""
        lines = [f"flowchart {direction}"]

        # Add nodes
        for node in graph_data.nodes:
            # Determine shape based on node type
            if node.node_type == "method":
                shape = f"[{node.label}]"
            elif node.node_type == "dataset":
                shape = f"[/{node.label}/]"
            else:
                shape = f"[{node.label}]"

            # Add year as subtext if available
            if node.year:
                lines.append(f'    {node.id}{shape}["{node.label} ({node.year})"]')
            else:
                lines.append(f'    {node.id}{shape}')

        # Add edges
        for edge in graph_data.edges:
            if edge.edge_type == "cites":
                lines.append(f"    {edge.source} --> {edge.target}")
            elif edge.edge_type == "extends":
                lines.append(f"    {edge.source} -.-> {edge.target}")
            elif edge.edge_type == "uses":
                lines.append(f"    {edge.source} ==>) {edge.target}")

        return "\n".join(lines)

    def generate_html(
        self,
        graph_data: CitationGraphData,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate interactive HTML visualization using D3.js."""
        html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>引用图谱 - Citation Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        #graph {
            width: 100%;
            height: 800px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .node circle {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .node circle:hover {
            stroke: #333;
            stroke-width: 3px;
        }
        .node text {
            font-size: 12px;
            fill: #333;
            pointer-events: none;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .tooltip {
            position: absolute;
            background: white;
            padding: 12px;
            border-radius: 6px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            max-width: 300px;
            font-size: 13px;
            line-height: 1.5;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .tooltip.visible {
            opacity: 1;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 论文引用图谱</h1>
        <div id="graph"></div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4CAF50;"></div>
                <span>本文引用</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2196F3;"></div>
                <span>方法</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FF9800;"></div>
                <span>数据集</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #9C27B0;"></div>
                <span>其他</span>
            </div>
        </div>
    </div>
    <div class="tooltip" id="tooltip"></div>

    <script>
        const GRAPH_DATA = __GRAPH_DATA__;

        const width = document.getElementById('graph').clientWidth;
        const height = 800;

        const colorScale = {
            'paper': '#4CAF50',
            'method': '#2196F3',
            'dataset': '#FF9800',
            'other': '#9C27B0'
        };

        const svg = d3.select('#graph')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g');

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });

        svg.call(zoom);

        // Force simulation
        const simulation = d3.forceSimulation(GRAPH_DATA.nodes)
            .force('link', d3.forceLink(GRAPH_DATA.edges).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(50));

        // Draw links
        const link = g.append('g')
            .selectAll('line')
            .data(GRAPH_DATA.edges)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', 2);

        // Draw nodes
        const node = g.append('g')
            .selectAll('g')
            .data(GRAPH_DATA.nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        node.append('circle')
            .attr('r', d => d.cited_count > 10 ? 20 : 12)
            .attr('fill', d => colorScale[d.node_type] || colorScale.other);

        node.append('text')
            .attr('dx', 15)
            .attr('dy', 4)
            .text(d => d.label.length > 20 ? d.label.substring(0, 20) + '...' : d.label);

        // Tooltip
        const tooltip = d3.select('#tooltip');

        node.on('mouseover', (event, d) => {
            tooltip.classed('visible', true)
                .html(`
                    <strong>${d.label}</strong><br>
                    ${d.year ? `年份: ${d.year}<br>` : ''}
                    ${d.authors ? `作者: ${d.authors}<br>` : ''}
                    ${d.title ? `标题: ${d.title.substring(0, 100)}...` : ''}
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', () => {
            tooltip.classed('visible', false);
        });

        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Initial zoom to fit
        svg.call(zoom.transform, d3.zoomIdentity.translate(width/4, height/4).scale(0.8));
    </script>
</body>
</html>
"""

        # Prepare data for JSON
        graph_json = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "year": node.year,
                    "authors": node.authors,
                    "title": node.title,
                    "node_type": node.node_type,
                    "cited_count": node.cited_count,
                }
                for node in graph_data.nodes
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type}
                for edge in graph_data.edges
            ],
        }

        html_content = html_template.replace("__GRAPH_DATA__", json.dumps(graph_json))

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Citation graph saved to {output_path}")

        return html_content

    def build_graph_from_tracker(
        self,
        citation_tracker,
        root_paper_title: str = "本文",
    ) -> CitationGraphData:
        """Build citation graph from CitationTracker."""
        graph = CitationGraphData(root_paper=root_paper_title)

        # Add root paper
        root_id = "root"
        graph.nodes.append(
            CitationNode(
                id=root_id,
                label=root_paper_title,
                node_type="paper",
            )
        )

        # Track existing node IDs to avoid duplicates
        existing_ids = {root_id}

        # Get all citations as list
        all_citations = citation_tracker.list_citations()

        # Add cited papers
        for citation in all_citations:
            ref_id = citation.reference_number
            # Create label from authors and year
            label = f"{citation.authors} ({citation.year})" if citation.authors and citation.year else ref_id

            if ref_id not in existing_ids:
                graph.nodes.append(
                    CitationNode(
                        id=ref_id,
                        label=label,
                        year=int(citation.year) if citation.year and citation.year.isdigit() else None,
                        authors=citation.authors or "",
                        title=citation.title or "",
                        node_type="paper",
                    )
                )
                existing_ids.add(ref_id)

            # Add edge from root to cited paper
            graph.edges.append(
                CitationEdge(source=root_id, target=ref_id, edge_type="cites")
            )

        return graph

    def save_graph_data(
        self,
        graph_data: CitationGraphData,
        output_path: Path,
    ) -> None:
        """Save graph data as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "year": node.year,
                    "authors": node.authors,
                    "title": node.title,
                    "node_type": node.node_type,
                    "cited_count": node.cited_count,
                }
                for node in graph_data.nodes
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type}
                for edge in graph_data.edges
            ],
            "root_paper": graph_data.root_paper,
        }

        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Graph data saved to {output_path}")
