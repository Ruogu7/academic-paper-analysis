"""Main paper agent orchestrator."""

import re
from pathlib import Path
from typing import Optional

from loguru import logger

from ..analysis import CitationTracker, CitationAnalyzer, ResultExtractor
from ..llm import LLMClient, Message
from ..output import MarkdownGenerator, PaperSummary
from ..parsers import PDFParser, UniversalParser
from ..prompts import (
    FINAL_SUMMARY_PROMPT,
    INNOVATION_PROMPT,
    PAPER_INTRO_PROMPT,
    PROJECT_CONTEXT_PROMPT,
    RESULTS_PROMPT,
    SYSTEM_PROMPT,
    TECHNIQUE_PROMPT,
    TITLE_AUTHORS_EXTRACT_PROMPT,
)
from ..config import settings
from .github_integration import GitHubIntegration, build_prompt_with_context


class PaperAgent:
    """Main agent for paper reading and summarization."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.llm = LLMClient(model=model, temperature=temperature)
        self.markdown_gen = MarkdownGenerator()
        self.citation_tracker = CitationTracker()
        self.citation_analyzer = CitationAnalyzer(llm_client=self.llm)
        self.result_extractor = ResultExtractor()
        self.github = GitHubIntegration()

    def read_and_summarize(
        self,
        paper_path: str | Path,
        output_path: Optional[Path] = None,
    ) -> PaperSummary:
        """Read a paper (PDF or HTML) and generate summary."""
        paper_path = Path(paper_path)

        logger.info(f"Starting paper analysis: {paper_path.name}")

        # Determine file type and parse accordingly
        suffix = paper_path.suffix.lower()

        if suffix in [".html", ".htm"]:
            # Parse HTML
            logger.info("Step 1: Parsing HTML...")
            parser = UniversalParser()
            parsed = parser.parse(paper_path)

            title = parsed.title if parsed.title != "Untitled" else paper_path.stem
            authors = parsed.metadata.get("author", "Unknown")
            full_text = parsed.text
        else:
            # Parse PDF (default)
            logger.info("Step 1: Parsing PDF...")
            parser = PDFParser(paper_path)
            doc = parser.parse()

            # Extract metadata - fallback to parsing first page if PDF metadata is empty
            title = doc.metadata.title if doc.metadata and doc.metadata.title != "Unknown Title" else paper_path.stem
            authors = doc.metadata.author if doc.metadata and doc.metadata.author != "Unknown Author" else "Unknown"

            # If authors still unknown, try to extract from first page using LLM
            if authors == "Unknown" and doc.pages:
                first_page_text = doc.pages[0].text[:2000]  # Increase text for more context
                title, authors = self._extract_title_authors_with_llm(
                    first_page_text, title, paper_path.stem
                )

            # Extract full text
            full_text = parser.get_full_text()

        # Step 3: Extract citations and results (pre-analysis)
        logger.info("Step 2: Extracting citations and results...")
        self.citation_tracker.extract_citations(full_text)
        self.result_extractor.extract_results(full_text)

        # Step 4: Extract GitHub URLs and fetch project context
        project_context = ""
        github_urls = self.github.extract_github_urls(full_text)
        if github_urls:
            logger.info(f"Found GitHub URLs: {github_urls}")
            project_context = self.github.fetch_project_context(github_urls)
            if project_context:
                logger.info("Fetched project context from GitHub")

        # Step 5: Generate structured summary via LLM
        logger.info("Step 3: Generating summary via LLM...")

        # For long papers, we may need to chunk
        if len(full_text) > 50000:
            logger.info("Paper is long, using chunked approach...")
            summary = self._summarize_long_paper(title, authors, full_text, project_context)
        else:
            summary = self._summarize_paper(title, authors, full_text, project_context)

        # Set metadata
        summary.source_file = str(paper_path)

        # Step 6: Analyze citations for relevance
        logger.info("Step 4: Analyzing citations...")
        citations = self.citation_tracker.list_citations()
        if citations:
            citation_analysis = self.citation_analyzer.analyze(
                paper_title=title,
                citations=citations,
                paper_content=full_text[:10000],  # Use first 10k chars as context
                relevance_threshold=0.5,
            )
            # Format citation analysis for the summary
            if citation_analysis.relevant_citations:
                summary.citation_analysis = self._format_citation_analysis(citation_analysis)

        # Step 5: Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            logger.info(f"Step 4: Saving to {output_path}")
            if not output_path.suffix:
                output_path = output_path / f"{paper_path.stem}_summary.md"
            self.markdown_gen.save(summary, output_path)

        logger.info(f"Summary generated: {title}")
        return summary

    def _summarize_paper(
        self,
        title: str,
        authors: str,
        content: str,
        project_context: str = "",
    ) -> PaperSummary:
        """Summarize a standard-length paper."""
        # Build the prompt with all content
        if project_context:
            prompt_content = build_prompt_with_context(
                title, authors, content, project_context
            )
        else:
            prompt_content = f"""## 论文信息
标题: {title}
作者: {authors}

## 论文内容
{content}

请根据以上内容，生成完整的论文总结。"""

        # Call LLM with final summary prompt
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt_content)],
            system_prompt=SYSTEM_PROMPT if not project_context else SYSTEM_PROMPT + "\n\n" + PROJECT_CONTEXT_PROMPT,
        )

        # Parse the structured response
        return self._parse_llm_response(response.content, title, authors)

    def _summarize_long_paper(
        self,
        title: str,
        authors: str,
        content: str,
        project_context: str = "",
    ) -> PaperSummary:
        """Summarize a long paper by chunking."""
        # Split into chunks
        max_chars = 40000
        chunks = []
        for i in range(0, len(content), max_chars):
            chunks.append(content[i : i + max_chars])

        logger.info(f"Processing {len(chunks)} chunks...")

        # Process each chunk for different aspects
        introduction = ""
        innovations = []
        technique = ""
        results = ""

        system_prompt = SYSTEM_PROMPT
        if project_context:
            system_prompt = SYSTEM_PROMPT + "\n\n" + PROJECT_CONTEXT_PROMPT

        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks for speed
            logger.info(f"Processing chunk {i + 1}/{len(chunks[:3])}...")

            # Introduction from first chunk
            if i == 0:
                response = self.llm.complete(
                    messages=[Message(role="user", content=PAPER_INTRO_PROMPT + "\n\n" + chunk)],
                    system_prompt=system_prompt,
                )
                introduction = response.content

            # Innovations from first two chunks
            if i < 2:
                response = self.llm.complete(
                    messages=[Message(role="user", content=INNOVATION_PROMPT + "\n\n" + chunk)],
                    system_prompt=system_prompt,
                )
                innovations.append(response.content)

            # Technique from any chunk
            response = self.llm.complete(
                messages=[Message(role="user", content=TECHNIQUE_PROMPT + "\n\n" + chunk)],
                system_prompt=system_prompt,
            )
            if not technique:
                technique = response.content

            # Results from later chunks
            if i >= 1:
                response = self.llm.complete(
                    messages=[Message(role="user", content=RESULTS_PROMPT + "\n\n" + chunk)],
                    system_prompt=SYSTEM_PROMPT,
                )
                if not results:
                    results = response.content

        # Generate final summary
        final_content = f"""
## 论文信息
标题: {title}
作者: {authors}

## 论文简介
{introduction}

## 论文创新点
{chr(10).join(innovations)}

## 核心技术
{technique}

## 实验结果
{results}
"""

        response = self.llm.complete(
            messages=[Message(role="user", content=FINAL_SUMMARY_PROMPT.format(
                title=title,
                authors=authors,
                content=final_content,
            ))],
            system_prompt=SYSTEM_PROMPT,
        )

        return self._parse_llm_response(response.content, title, authors)

    def _parse_llm_response(self, content: str, title: str, authors: str) -> PaperSummary:
        """Parse LLM response into structured summary."""
        summary = PaperSummary(
            title=title,
            authors=authors,
        )

        # Initialize section tracking
        sections = {
            "abstract": [],
            "introduction": [],
            "innovations": [],
            "technique": [],
            "method_details": [],
            "experiments": [],
            "results": [],
            "discussion": [],
            "conclusion": [],
            "limitations": [],
            "future_work": [],
        }
        current_section = None
        section_content = []

        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers (check for markdown headers ## or ###)
            section_key = self._detect_section(line)
            if section_key:
                # Save previous section content
                if current_section and section_content:
                    sections[current_section].extend(section_content)
                # Start new section
                current_section = section_key
                section_content = []
                # Remove the header line from content
                continue

            # Add content to current section
            if current_section:
                section_content.append(line)
            else:
                # If no section detected yet, collect content
                section_content.append(line)

        # Save last section
        if current_section and section_content:
            sections[current_section].extend(section_content)

        # Assign to summary fields
        summary.abstract = "\n".join(sections["abstract"])
        summary.introduction = "\n".join(sections["introduction"])

        # Parse innovations - could be list items or paragraphs
        if sections["innovations"]:
            summary.innovations = self._parse_innovations(sections["innovations"])

        summary.technique = "\n".join(sections["technique"])
        summary.method_details = "\n".join(sections["method_details"])
        summary.experiments = "\n".join(sections["experiments"])
        summary.results = "\n".join(sections["results"])
        summary.discussion = "\n".join(sections["discussion"])
        summary.conclusion = "\n".join(sections["conclusion"])
        summary.limitations = "\n".join(sections["limitations"])
        summary.future_work = "\n".join(sections["future_work"])

        # Fallback: if results is empty but we have unassigned content
        if not summary.results and sections["abstract"]:
            # Try to use abstract as results fallback
            pass

        return summary

    def _detect_section(self, line: str) -> Optional[str]:
        """Detect section type from header line."""
        # Remove markdown headers (# ## ###)
        clean_line = line.lstrip("#").strip().lower()

        # Abstract section
        if "摘要" in clean_line or "abstract" in clean_line:
            return "abstract"

        # Introduction section
        if "简介" in clean_line or "引言" in clean_line or "introduction" in clean_line:
            return "introduction"

        # Innovation section
        if "创新" in clean_line or "contribution" in clean_line or "innovation" in clean_line:
            return "innovations"

        # Technique/Method section
        if "技术" in clean_line or "方法" in clean_line or "technique" in clean_line or "method" in clean_line:
            return "technique"

        # Method details
        if "细节" in clean_line or "details" in clean_line or "实现" in clean_line:
            return "method_details"

        # Experiments
        if "实验" in clean_line or "experiment" in clean_line:
            return "experiments"

        # Results
        if "结果" in clean_line or "result" in clean_line:
            return "results"

        # Discussion
        if "讨论" in clean_line or "discussion" in clean_line:
            return "discussion"

        # Conclusion
        if "结论" in clean_line or "conclusion" in clean_line:
            return "conclusion"

        # Limitations
        if "局限" in clean_line or "limitation" in clean_line:
            return "limitations"

        # Future work
        if "未来" in clean_line or "future" in clean_line:
            return "future_work"

        return None

    def _parse_innovations(self, content: list[str]) -> list[str]:
        """Parse innovation content into structured list."""
        innovations = []
        current_innovation = []

        for line in content:
            # Check for numbered items (1. 2. 3. or - )
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                if current_innovation:
                    innovations.append("\n".join(current_innovation))
                    current_innovation = []
            current_innovation.append(line)

        if current_innovation:
            innovations.append("\n".join(current_innovation))

        return innovations if innovations else ["\n".join(content)]

    def _extract_title_authors_from_text(
        self, text: str, current_title: str, fallback_title: str
    ) -> tuple[str, str]:
        """Extract title and authors from first page text.

        Args:
            text: First page text content
            current_title: Current title from PDF metadata or filename
            fallback_title: Fallback title from filename

        Returns:
            Tuple of (title, authors)
        """
        import re

        lines = text.split("\n")
        title = current_title
        authors = "Unknown"

        # Patterns to exclude from title
        exclude_patterns = [
            r"^abstract$",
            r"^introduction$",
            r"^\d+$",
            r"^chapter\s+\d+",
            r"^section\s+\d+",
        ]

        # Patterns that indicate authors line
        author_indicators = [
            "university", "institute", "college", "school", "department",
            "lab", "laboratory", "research", "@", "email", "tech",
        ]

        # Find title: usually the first non-empty line that's not too short
        title_candidates = []
        for line in lines[:20]:
            line = line.strip()
            if not line:
                continue
            # Skip if too short or matches exclude patterns
            if len(line) < 15:
                continue
            if any(re.match(p, line.lower()) for p in exclude_patterns):
                continue
            # Title should not contain typical author indicators
            if any(ind in line.lower() for ind in author_indicators):
                continue
            title_candidates.append(line)
            if len(title_candidates) >= 3:
                break

        # Use first significant line as title if current title is just filename
        if title == fallback_title and title_candidates:
            # Check if the candidate looks like a good title
            first_candidate = title_candidates[0]
            # Good titles are usually longer and don't end with typical author affiliations
            if len(first_candidate) > 20 and not any(ind in first_candidate.lower() for ind in author_indicators):
                title = first_candidate

        # Find authors: look for lines after title that contain author indicators
        in_author_section = False
        author_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                in_author_section = False
                continue

            # If we found title candidates, start looking for authors after them
            if title_candidates and any(line == tc for tc in title_candidates[:2]):
                in_author_section = True
                continue

            if in_author_section:
                # Check if this line looks like authors
                if len(line) > 10 and len(line) < 200:
                    # Check for author indicators
                    if any(ind in line.lower() for ind in author_indicators) or "," in line or "&" in line:
                        author_lines.append(line)
                    elif len(author_lines) > 0 and line[0].isupper():
                        # Could be continuation of authors
                        author_lines.append(line)
                    elif author_lines:
                        # Probably moved to next section
                        break

        if author_lines:
            authors = " ".join(author_lines[:2])  # Take first 2 lines max

        return title, authors

    def _extract_title_authors_with_llm(
        self, text: str, current_title: str, fallback_title: str
    ) -> tuple[str, str]:
        """Extract title and authors using LLM.

        Args:
            text: First page text content
            current_title: Current title from PDF metadata
            fallback_title: Fallback title from filename

        Returns:
            Tuple of (title, authors)
        """
        import json

        # Build prompt
        prompt = TITLE_AUTHORS_EXTRACT_PROMPT.format(first_page_text=text[:3000])

        try:
            response = self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                system_prompt="你是一个学术论文信息提取助手。请准确提取论文的标题和作者信息。",
            )

            # Parse JSON response
            content = response.content.strip()
            # Handle possible markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())

            title = data.get("title") or current_title or fallback_title
            authors = data.get("authors") or "Unknown"

            logger.info(f"LLM extracted title: {title[:50]}..., authors: {authors[:50]}...")

            return title, authors

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to regex method")
            # Fallback to original method on failure
            return self._extract_title_authors_from_text(text, current_title, fallback_title)

    def _format_citation_analysis(self, analysis) -> str:
        """Format citation analysis into markdown string.

        Args:
            analysis: CitationAnalysis object

        Returns:
            Formatted markdown string
        """
        lines = [
            "## 相关引文分析",
            "",
            f"从 {analysis.total_citations} 篇引文中，分析出 {len(analysis.relevant_citations)} 篇与论文内容高度相关的文献：",
            "",
        ]

        for citation in analysis.relevant_citations:
            # 标题放上面
            title = citation.title if citation.title and citation.title != "Unknown" else f"{citation.first_author} et al., {citation.year}"
            lines.append(f"### {citation.reference_number} {title}")
            lines.append("")
            # 作者和年份
            lines.append(f"- **作者**: {citation.first_author}, {citation.year}")
            # 添加链接
            if citation.paper_link:
                # 清理链接中的空格和多余字符
                link = citation.paper_link.strip()
                # 移除所有类型的空白字符，包括全角空格
                link = link.replace(' ', '').replace('\u3000', '').replace('\t', '')

                # 尝试提取 arXiv ID（格式如 2402.03069）
                arxiv_match = re.search(r'(\d{4}\.\d{4,5})', link)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                    lines.append(f"- **链接**: [{arxiv_url}]({arxiv_url})")
                elif "github.com" in link.lower() or "github:" in link.lower():
                    # GitHub 链接 - 清理并构建
                    github_match = re.search(r'github\.com[/:]([^\s]+)', link)
                    if github_match:
                        github_url = f"https://github.com/{github_match.group(1)}"
                        lines.append(f"- **链接**: [{github_url}]({github_url})")
                    else:
                        lines.append(f"- **链接**: {link}")
                elif link.startswith("http"):
                    lines.append(f"- **链接**: [{link}]({link})")
                else:
                    lines.append(f"- **链接**: {link}")
            lines.append(f"- **主题**: {citation.topic}")
            lines.append(f"- **相关性**: {citation.relevance_score:.2f}")
            lines.append(f"- **贡献**: {citation.contribution}")
            lines.append("")

        return "\n".join(lines)
