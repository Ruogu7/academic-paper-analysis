"""Markdown generator for paper summaries."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..llm import LLMClient, Message
from ..prompts import MARKDOWN_FORMAT_CHECK_PROMPT
from ..config import settings


@dataclass
class PaperSummary:
    """Complete paper summary data."""

    title: str
    authors: str
    institution: Optional[str] = None
    publication_venue: Optional[str] = None
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

    abstract: str = ""
    introduction: str = ""
    innovations: list[str] = field(default_factory=list)
    technique: str = ""
    method_details: str = ""
    experiments: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    limitations: str = ""
    future_work: str = ""
    references: list[str] = field(default_factory=list)

    # Extracted data
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)

    # Citation analysis
    citation_analysis: Optional[str] = None

    source_file: Optional[str] = None
    generated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.innovations is None:
            self.innovations = []
        if self.references is None:
            self.references = []
        if self.generated_at is None:
            self.generated_at = datetime.now()


class MarkdownGenerator:
    """Generate Markdown summary from paper analysis."""

    def generate(self, summary: PaperSummary) -> str:
        """Generate Markdown content from summary data."""
        parts = []

        # Header
        parts.append(f"# {summary.title}")
        parts.append("")

        # Metadata
        parts.extend(self._generate_metadata(summary))

        # Abstract
        if summary.abstract:
            parts.extend(self._generate_section("摘要", summary.abstract))

        # Introduction
        if summary.introduction:
            parts.extend(self._generate_section("论文简介", summary.introduction))

        # Innovations
        if summary.innovations:
            parts.extend(self._generate_innovations(summary.innovations))

        # Technique/Method
        if summary.technique:
            parts.extend(self._generate_section("核心技术", summary.technique))

        if summary.method_details:
            parts.extend(self._generate_section("方法细节", summary.method_details))

        # Experiments
        if summary.experiments:
            parts.extend(self._generate_section("实验设置", summary.experiments))

        # Results
        if summary.results:
            parts.extend(self._generate_section("实验结果", summary.results))

        # Discussion
        if summary.discussion:
            parts.extend(self._generate_section("讨论", summary.discussion))

        # Conclusion
        if summary.conclusion:
            parts.extend(self._generate_section("结论", summary.conclusion))

        # Limitations and Future Work
        if summary.limitations or summary.future_work:
            parts.extend(self._generate_section("局限性与未来工作",
                f"{summary.limitations}\n\n**未来工作**\n{summary.future_work}"))

        # Extracted Data
        parts.extend(self._generate_extracted_data(summary))

        # References
        if summary.references:
            parts.extend(self._generate_references(summary.references))

        # Citation Analysis
        if summary.citation_analysis:
            parts.append("")
            parts.append(summary.citation_analysis)

        return "\n".join(parts)

    def _generate_metadata(self, summary: PaperSummary) -> list[str]:
        """Generate metadata section."""
        lines = ["## 论文信息", ""]
        if summary.authors:
            lines.append(f"- **作者**: {summary.authors}")
        if summary.institution:
            lines.append(f"- **机构**: {summary.institution}")
        if summary.publication_venue:
            lines.append(f"- **发表场所**: {summary.publication_venue}")
        if summary.year:
            lines.append(f"- **年份**: {summary.year}")
        if summary.arxiv_id:
            lines.append(f"- **arXiv**: [{summary.arxiv_id}](https://arxiv.org/abs/{summary.arxiv_id})")
        if summary.github:
            lines.append(f"- **GitHub**: [{summary.github}]({summary.github})")
        if summary.website:
            lines.append(f"- **项目网站**: [{summary.website}]({summary.website})")
        if summary.source_file:
            lines.append(f"- **源文件**: `{summary.source_file}`")
        lines.append(f"- **生成时间**: {summary.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        return lines

    def _generate_section(self, title: str, content: str) -> list[str]:
        """Generate a section with title and content."""
        lines = [f"## {title}", ""]
        lines.append(content)
        lines.append("")
        return lines

    def _generate_innovations(self, innovations: list[str]) -> list[str]:
        """Generate innovations section."""
        lines = ["## 论文创新点", ""]
        for i, innovation in enumerate(innovations, 1):
            lines.append(f"### {i}. {innovation.get('title', '创新点')}" if isinstance(innovation, dict) else f"{i}. {innovation}")
            if isinstance(innovation, dict) and 'description' in innovation:
                lines.append("")
                lines.append(innovation['description'])
            lines.append("")
        return lines

    def _generate_extracted_data(self, summary: PaperSummary) -> list[str]:
        """Generate extracted data section."""
        lines = []

        # Datasets
        if summary.datasets:
            lines.append("## 数据集")
            lines.append("")
            for dataset in summary.datasets:
                lines.append(f"- {dataset}")
            lines.append("")

        # Metrics
        if summary.metrics:
            lines.append("## 评估指标")
            lines.append("")
            for metric in summary.metrics:
                lines.append(f"- {metric}")
            lines.append("")

        # Tables
        if summary.tables:
            lines.append("## 表格")
            lines.append("")
            for table_id, description in summary.tables.items():
                lines.append(f"### Table {table_id}")
                lines.append("")
                lines.append(description[:500])  # Truncate long descriptions
                lines.append("")

        # Figures
        if summary.figures:
            lines.append("## 图表")
            lines.append("")
            for fig_id, description in summary.figures.items():
                lines.append(f"### Figure {fig_id}")
                lines.append("")
                lines.append(description[:500])
                lines.append("")

        return lines

    def _generate_references(self, references: list[str]) -> list[str]:
        """Generate references section."""
        lines = ["## 参考文献", ""]
        for i, ref in enumerate(references, 1):
            lines.append(f"{i}. {ref}")
        lines.append("")
        return lines

    def _validate_and_fix_format(self, content: str) -> str:
        """Validate and fix Markdown format using LLM.

        Args:
            content: Original Markdown content

        Returns:
            Fixed Markdown content
        """
        try:
            # Use lightweight model for format validation
            llm = LLMClient(model=settings.default_model, temperature=0.3)

            prompt = MARKDOWN_FORMAT_CHECK_PROMPT.format(markdown_content=content)

            response = llm.complete(
                messages=[Message(role="user", content=prompt)],
                system_prompt="你是一个Markdown格式专家。请检查并修复Markdown文档的格式问题。",
            )

            fixed_content = response.content.strip()

            # Verify returned content is valid
            if fixed_content and len(fixed_content) > len(content) * 0.5:
                logger.info("Markdown format validated and fixed by LLM")
                return fixed_content
            else:
                logger.warning("LLM format fix returned invalid content, using original")
                return content

        except Exception as e:
            logger.warning(f"Markdown format validation failed: {e}, using original content")
            return content

    def _add_spacing_around_markdown_symbols(self, content: str) -> str:
        """Add spaces around Markdown inline symbols for better rendering.

        Args:
            content: Original Markdown content

        Returns:
            Content with spaces around Markdown symbols
        """
        # Strategy: Use placeholders to avoid nested matching
        # 1. First handle bold italic (***text***)
        # 2. Then handle bold (**text**)
        # 3. Then handle italic (*text*)
        # 4. Finally handle strikethrough and code

        # Use a temporary placeholder to protect already-processed markers
        # We'll use a technique: match and replace with unique placeholders, then restore

        result = content

        # Bold italic ***text*** - must have exactly 3 stars on each side
        # Use (?=...) lookahead to ensure we don't match if there's more stars
        result = re.sub(r'\*\*\*(?=[^*])(.+?)(?<=[^*])\*\*\*', r' ___BOLDITALIC_START\1BOLDITALIC_END___ ', result)

        # Bold **text** - exactly 2 stars (not followed by more stars)
        result = re.sub(r'\*\*(?=[^*])(.+?)(?<=[^*])\*\*', r' ___BOLD_START\1BOLD_END___ ', result)

        # Italic *text* - exactly 1 star (not part of ** or ***)
        result = re.sub(r'(?<![*])\*(?=[^*])(.+?)(?<=[^*])\*(?![*])', r' ___ITALIC_START\1ITALIC_END___ ', result)

        # Strikethrough ~~text~~
        result = re.sub(r'~~(.+?)~~', r' ___STRIKETHROUGH_START\1STRIKETHROUGH_END___ ', result)

        # Inline code `code`
        result = re.sub(r'`(.+?)`', r' ___CODE_START\1CODE_END___ ', result)

        # Now add spaces around the placeholders
        result = result.replace('___BOLDITALIC_START', '***')
        result = result.replace('BOLDITALIC_END___', '***')

        result = result.replace('___BOLD_START', '**')
        result = result.replace('BOLD_END___', '**')

        result = result.replace('___ITALIC_START', '*')
        result = result.replace('ITALIC_END___', '*')

        result = result.replace('___STRIKETHROUGH_START', '~~')
        result = result.replace('STRIKETHROUGH_END___', '~~')

        result = result.replace('___CODE_START', '`')
        result = result.replace('CODE_END___', '`')

        # Clean up multiple spaces
        result = re.sub(r' +', ' ', result)

        return result

    def _add_spacing_between_chinese_english(self, content: str) -> str:
        """Add spaces between Chinese and English text for better readability.

        Args:
            content: Original content

        Returns:
            Content with spaces between Chinese and English
        """
        # Chinese character range: \u4e00-\u9fff (common), plus \u3400-\u4dbf (extension A)
        chinese_chars = '\u4e00-\u9fff\u3400-\u4dbf'

        # Strategy: Protect code blocks and tables first, then process, then restore
        # Use placeholders to protect these sections

        # Protect markdown links: [text](url)
        markdown_links = []
        def protect_markdown_link(m):
            markdown_links.append(m.group(0))
            return f'__MDLINK_{len(markdown_links) - 1}__'
        result = re.sub(r'\[[^\]]+\]\([^)]+\)', protect_markdown_link, content)

        # Protect code blocks (```...```)
        code_blocks = []
        def protect_code_block(m):
            code_blocks.append(m.group(0))
            return f'__CODEBLOCK_{len(code_blocks) - 1}__'
        result = re.sub(r'```[\s\S]*?```', protect_code_block, result)

        # Protect inline code (`...`)
        inline_codes = []
        def protect_inline_code(m):
            inline_codes.append(m.group(0))
            return f'__INLINECODE_{len(inline_codes) - 1}__'
        result = re.sub(r'`[^`]+`', protect_inline_code, result)

        # Protect tables (|...| rows)
        tables = []
        def protect_table(m):
            tables.append(m.group(0))
            return f'__TABLE_{len(tables) - 1}__'
        result = re.sub(r'(\|.+\|\n?)+', protect_table, result)

        # Now process the remaining content
        # First, add spaces around operators (= + * / % ^ |)
        # This handles cases like "d=0.5" → "d = 0.5"
        # Note: We don't handle '-' here because it's used in model names like "GPT-3.5"
        operators = ['=', '+', '*', '/', '%', '^', '|']
        for op in operators:
            # Add space around operator between digits
            result = re.sub(rf'(\d+)\s*{re.escape(op)}\s*(\d+)', rf'\1 {op} \2', result)
            # Handle word/digit followed by operator followed by word/digit
            result = re.sub(rf'([a-zA-Z0-9]+)\s*{re.escape(op)}\s*([a-zA-Z0-9])', rf'\1 {op} \2', result)

        # Add space between Chinese and English words (not characters)
        # Pattern: Chinese character followed by English word
        result = re.sub(f'([{chinese_chars}])([a-zA-Z0-9]+)', r'\1 \2', result)

        # Add space between English word and Chinese character
        result = re.sub(f'([a-zA-Z0-9]+)([{chinese_chars}])', r'\1 \2', result)

        # Add space after punctuation before English (but not after each character)
        # First, handle Chinese punctuation
        result = re.sub(r'([,，.。!！?？;；:：])([a-zA-Z])', r'\1 \2', result)

        # Add space before opening parenthesis after Chinese
        result = re.sub(f'([{chinese_chars}])\\(', r'\1 (', result)

        # Add space after closing parenthesis before Chinese
        result = re.sub(f'\\)([{chinese_chars}])', r') \1', result)

        # Add space after opening bracket before Chinese
        result = re.sub(f'\\[([{chinese_chars}])', r'[ \1', result)

        # Add space before closing bracket after Chinese
        result = re.sub(f'([{chinese_chars}])\\]', r'\1 ]', result)

        # Restore protected sections
        for i, block in enumerate(code_blocks):
            result = result.replace(f'__CODEBLOCK_{i}__', block)
        for i, code in enumerate(inline_codes):
            result = result.replace(f'__INLINECODE_{i}__', code)
        for i, table in enumerate(tables):
            result = result.replace(f'__TABLE_{i}__', table)
        for i, link in enumerate(markdown_links):
            result = result.replace(f'__MDLINK_{i}__', link)

        # Clean up multiple spaces
        result = re.sub(r' +', ' ', result)

        return result

    def save(self, summary: PaperSummary, output_path: Path) -> Path:
        """Generate and save Markdown to file."""
        content = self.generate(summary)

        # Validate and fix format using LLM
        content = self._validate_and_fix_format(content)

        # Add spacing fixes for better Markdown rendering
        content = self._add_spacing_around_markdown_symbols(content)
        content = self._add_spacing_between_chinese_english(content)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def save_with_figures(
        self,
        summary: PaperSummary,
        output_path: Path,
        figures: Optional[list[tuple[Path, str]]] = None,
    ) -> Path:
        """Generate and save Markdown with embedded figures."""
        content = self.generate(summary)

        # Append figures section if provided
        if figures:
            content += "\n## 图表\n"
            for fig_path, caption in figures:
                fig_rel = fig_path.name
                content += f"\n### {caption}\n"
                content += f"![{caption}]({fig_rel})\n"

        # Validate and fix format using LLM
        content = self._validate_and_fix_format(content)

        # Add spacing fixes for better Markdown rendering
        content = self._add_spacing_around_markdown_symbols(content)
        content = self._add_spacing_between_chinese_english(content)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        # Copy figures to output directory
        if figures:
            fig_dir = output_path.parent / "figures"
            fig_dir.mkdir(exist_ok=True)
            for fig_path, _ in figures:
                import shutil

                shutil.copy(fig_path, fig_dir / fig_path.name)

        return output_path
