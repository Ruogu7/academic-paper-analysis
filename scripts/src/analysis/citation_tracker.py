"""Citation tracking for academic papers."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Citation:
    """Represents a citation in the paper."""

    reference_number: str
    cited_text: str
    context: str
    page_number: Optional[int] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None


@dataclass
class CitationGraph:
    """Graph of citation relationships."""

    citations: dict[str, Citation] = field(default_factory=dict)
    citation_count: dict[str, int] = field(default_factory=dict)  # ref_number -> count

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the graph."""
        # Handle multiple citations like [1,2,3]
        refs = re.split(r"[,\s]+", citation.reference_number)
        for ref in refs:
            ref = ref.strip()
            if ref:
                if ref not in self.citations:
                    self.citations[ref] = citation
                    self.citation_count[ref] = 0
                self.citation_count[ref] += 1


class CitationTracker:
    """Track and manage citations in academic papers."""

    # Common citation patterns
    CITATION_PATTERNS = [
        r"\[(\d+(?:,\s*\d+)*)\]",  # [1], [1,2,3]
        r"\(([\w\s]+(?:et\s+al\.?)?(?:&\s+[\w\s]+)?),\s*(\d{4})[a-z]?\)",  # (Smith et al., 2020)
        r"\(([\w\s]+),\s*(\d{4})[a-z]?\)",  # (Smith, 2020)
    ]

    def __init__(self):
        self.citations: dict[str, Citation] = {}
        self._patterns = [re.compile(p, re.MULTILINE) for p in self.CITATION_PATTERNS]
        self.citation_graph = CitationGraph()

    def extract_citations(self, text: str, page_number: Optional[int] = None) -> list[Citation]:
        """Extract all citations from given text."""
        found = []

        for pattern in self._patterns:
            matches = pattern.finditer(text)
            for match in matches:
                # Check if it's a numbered reference [1] or author-year (Smith, 2020)
                if match.lastindex == 1:
                    # Numbered citation [1] or [1,2,3]
                    reference_number = match.group(1)
                    authors = None
                    year = None
                elif match.lastindex == 2:
                    # Author-year citation with year as group 2
                    reference_number = f"{match.group(1).strip()}_{match.group(2)}"
                    authors = match.group(1).strip()
                    year = match.group(2)
                elif match.lastindex == 3:
                    # Author-year citation with author and year separate
                    reference_number = f"{match.group(1).strip()}_{match.group(2)}"
                    authors = match.group(1).strip()
                    year = match.group(2)
                else:
                    continue

                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)

                citation = Citation(
                    reference_number=reference_number,
                    cited_text=match.group(0),
                    context=f"...{text[context_start:match.start()]}[{match.group(0)}]{text[match.end():context_end]}...",
                    page_number=page_number,
                    authors=authors,
                    year=year,
                )
                found.append(citation)
                self.citations[reference_number] = citation
                self.citation_graph.add_citation(citation)

        return found

    def get_citation_context(self, reference_number: str) -> Optional[Citation]:
        """Get full context for a specific citation."""
        return self.citations.get(reference_number)

    def list_citations(self) -> list[Citation]:
        """List all extracted citations."""
        return list(self.citations.values())

    def get_most_cited(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get the most cited references."""
        sorted_citations = sorted(
            self.citation_graph.citation_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_citations[:top_n]

    def extract_references_section(self, full_text: str) -> str:
        """Extract the references section from the paper."""
        # Look for references section
        patterns = [
            r"(?i)^(?:references|参考文献)\s*\n(.*)",
            r"(?i)^[0-9]+\s+(?:references|参考文献)\s*\n(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.MULTILINE)
            if match:
                return match.group(1)

        # Fallback: return last 5000 characters
        return full_text[-5000:]
