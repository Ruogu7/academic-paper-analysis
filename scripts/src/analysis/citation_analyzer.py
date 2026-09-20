"""Citation analysis for academic papers."""

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from ..llm import LLMClient, Message


@dataclass
class RelevantCitation:
    """A citation relevant to the paper's content."""

    reference_number: str
    first_author: str
    year: str
    title: str  # 论文完整标题
    topic: str
    relevance_score: float
    contribution: str
    paper_link: str = ""  # 论文链接 (arXiv 或其他)
    context_in_paper: str = ""


@dataclass
class CitationAnalysis:
    """Analysis result of paper citations."""

    paper_title: str
    total_citations: int
    analyzed_citations: int
    relevant_citations: list[RelevantCitation] = field(default_factory=list)
    summary: str = ""


class CitationAnalyzer:
    """Analyze citations to identify relevant works."""

    # Prompt for analyzing citations
    CITATION_ANALYSIS_PROMPT = """你是一位学术论文分析专家。请分析以下论文中的引用文献，识别出与论文内容最相关的引文，并进行简要叙述分析。

论文标题: {paper_title}

论文内容摘要:
{paper_content}

引文列表:
{citations}

请对每个引文进行分析，识别出与论文最相关的前 {top_n} 个引文。

对于每个相关引文，请提供以下信息：
1. 引用编号 (如 [1], [2])
2. 第一作者姓名
3. 年份
4. 论文完整标题（必须）
5. 该文献的主题/研究领域
6. 与论文的相关性评分 (0-1之间的浮点数)
7. 该文献对论文的贡献/启发 (1-2句话)
8. 论文链接（arXiv链接、GitHub或其他可访问的链接，如果没有请标注"未知"）

请严格按照以下格式输出：

## 相关引文分析

### [引用编号] 作者, 年份
- **标题**: <论文完整标题>
- **第一作者**: <作者>
- **年份**: <年份>
- **主题**: <主题>
- **相关性**: <0-1之间的分数>
- **贡献**: <贡献描述>
- **链接**: <论文链接>"""

    def __init__(self, llm_client: Optional[LLMClient] = None, top_n: int = 5):
        """Initialize the citation analyzer.

        Args:
            llm_client: Optional LLM client, will create new one if not provided
            top_n: Number of top citations to analyze
        """
        self.llm = llm_client or LLMClient()
        self.top_n = top_n

    def analyze(
        self,
        paper_title: str,
        citations: list,
        paper_content: str,
        relevance_threshold: float = 0.5,
    ) -> CitationAnalysis:
        """Analyze citations and identify relevant ones.

        Args:
            paper_title: Title of the paper
            citations: List of Citation objects
            paper_content: Content of the paper for context
            relevance_threshold: Minimum relevance score to include

        Returns:
            CitationAnalysis object with relevant citations
        """
        if not citations:
            return CitationAnalysis(
                paper_title=paper_title,
                total_citations=0,
                analyzed_citations=0,
                relevant_citations=[],
            )

        # Prepare citations for analysis
        citations_text = self._prepare_citations_text(citations)

        # Build prompt
        prompt = self.CITATION_ANALYSIS_PROMPT.format(
            paper_title=paper_title,
            paper_content=paper_content[:5000],  # Limit content length
            citations=citations_text,
            top_n=self.top_n,
        )

        try:
            # Call LLM
            response = self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                system_prompt="你是一位学术论文分析专家，擅长分析文献引用关系。",
            )

            # Parse response
            relevant_citations = self._parse_llm_response(response.content)

            # Filter by threshold
            filtered_citations = [
                c for c in relevant_citations
                if c.relevance_score >= relevance_threshold
            ]

            logger.info(f"Analyzed {len(citations)} citations, found {len(filtered_citations)} relevant")

            return CitationAnalysis(
                paper_title=paper_title,
                total_citations=len(citations),
                analyzed_citations=len(relevant_citations),
                relevant_citations=filtered_citations,
                summary=response.content,
            )

        except Exception as e:
            logger.error(f"Error analyzing citations: {e}")
            return CitationAnalysis(
                paper_title=paper_title,
                total_citations=len(citations),
                analyzed_citations=0,
                relevant_citations=[],
            )

    def _prepare_citations_text(self, citations: list) -> str:
        """Prepare citations text for LLM prompt.

        Args:
            citations: List of Citation objects

        Returns:
            Formatted citations string
        """
        lines = []
        for i, citation in enumerate(citations[:20], 1):  # Limit to 20 citations
            ref_num = getattr(citation, 'reference_number', f'[{i}]')
            context = getattr(citation, 'context', '')[:200]
            cited_text = getattr(citation, 'cited_text', '')
            authors = getattr(citation, 'authors', None)
            year = getattr(citation, 'year', None)

            lines.append(f"{ref_num}: {cited_text}")
            if authors:
                lines.append(f"   作者: {authors}")
            if year:
                lines.append(f"   年份: {year}")
            lines.append(f"   上下文: {context}")
            lines.append("")

        return "\n".join(lines)

    def _parse_llm_response(self, content: str) -> list[RelevantCitation]:
        """Parse LLM response to extract relevant citations.

        Args:
            content: LLM response content

        Returns:
            List of RelevantCitation objects
        """
        citations = []

        # Remove markdown headers and bold markers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
        content = content.replace('**', '')

        # Split by lines and process
        lines = content.split('\n')

        current_citation = {}
        for line in lines:
            line = line.strip()

            # Check for new citation reference
            ref_match = re.search(r'\[(\d+)\]', line)
            if ref_match:
                # Save previous citation if exists
                if current_citation and 'reference_number' in current_citation:
                    citations.append(self._create_citation(current_citation))

                current_citation = {'reference_number': f"[{ref_match.group(1)}]"}

                # Also try to extract author and year from citation line like "[1] Smith et al., 2020"
                author_year_match = re.search(r'\[?\d+\]?\s*([A-Z][a-zA-Z]+).*?(\d{4})', line)
                if author_year_match:
                    current_citation['first_author'] = author_year_match.group(1)
                    current_citation['year'] = author_year_match.group(2)

            # Extract fields from current citation block
            if 'reference_number' in current_citation:
                # First author
                author_match = re.search(r'(?:第一作者|Author)[:\s]+([^\n,]+)', line)
                if author_match:
                    current_citation['first_author'] = author_match.group(1).strip()

                # Year
                year_match = re.search(r'(?:年份|Year)[:\s]+(\d{4})', line)
                if year_match:
                    current_citation['year'] = year_match.group(1)

                # Title (论文完整标题)
                title_match = re.search(r'(?:标题|Title)[:\s]+([^\n]+)', line)
                if title_match:
                    current_citation['title'] = title_match.group(1).strip()

                # Topic
                topic_match = re.search(r'(?:主题|Topic)[:\s]+([^\n]+)', line)
                if topic_match:
                    current_citation['topic'] = topic_match.group(1).strip()

                # Relevance score
                relevance_match = re.search(r'(?:相关性|Relevance)[:\s]+([0-9.]+)', line)
                if relevance_match:
                    try:
                        current_citation['relevance_score'] = float(relevance_match.group(1))
                    except ValueError:
                        pass

                # Contribution
                contribution_match = re.search(r'(?:贡献|Contribution)[:\s]+([^\n]+)', line)
                if contribution_match:
                    current_citation['contribution'] = contribution_match.group(1).strip()

                # Paper link
                link_match = re.search(r'(?:链接|Link|Paper)[:\s]+([^\n]+)', line)
                if link_match:
                    link = link_match.group(1).strip()
                    # Clean up common prefixes
                    link = re.sub(r'^(arxiv:|github:|http)', '', link, flags=re.IGNORECASE)
                    current_citation['paper_link'] = link

        # Don't forget the last citation
        if current_citation and 'reference_number' in current_citation:
            citations.append(self._create_citation(current_citation))

        return citations

    def _create_citation(self, data: dict) -> RelevantCitation:
        """Create a RelevantCitation from parsed data."""
        return RelevantCitation(
            reference_number=data.get('reference_number', '[?]'),
            first_author=data.get('first_author', 'Unknown'),
            year=data.get('year', 'Unknown'),
            title=data.get('title', 'Unknown'),
            topic=data.get('topic', 'Unknown'),
            relevance_score=data.get('relevance_score', 0.5),
            contribution=data.get('contribution', ''),
            paper_link=data.get('paper_link', ''),
        )
