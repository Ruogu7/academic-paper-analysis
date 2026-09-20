"""Section detector for academic papers."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SectionType(Enum):
    """Types of sections in academic papers."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    METHODOLOGY = "methodology"
    APPROACH = "approach"
    MODEL = "model"
    ALGORITHM = "algorithm"
    EXPERIMENT = "experiment"
    RESULTS = "results"
    EVALUATION = "evaluation"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """Represents a section in the paper."""

    section_type: SectionType
    title: str
    content: str
    start_page: int
    end_page: int
    level: int  # 1 for main sections, 2 for subsections


# Common section title patterns
SECTION_PATTERNS: dict[SectionType, list[str]] = {
    SectionType.ABSTRACT: [r"^abstract$", r"^摘要$", r"^1?\s*abstract"],
    SectionType.INTRODUCTION: [r"^1?\s*introduction$", r"^引言$", r"^1?\s*引言"],
    SectionType.RELATED_WORK: [r"^related\s*work$", r"^2?\s*related\s*work", r"^相关工作$"],
    SectionType.METHOD: [r"^2?\s*method", r"^方法$", r"^2?\s*方法"],
    SectionType.METHODOLOGY: [r"^2?\s*methodology", r"^方法论$"],
    SectionType.APPROACH: [r"^2?\s*approach", r"^方法与模型"],
    SectionType.MODEL: [r"^2?\s*model", r"^模型$", r"^2?\s*模型"],
    SectionType.ALGORITHM: [r"^2?\s*algorithm", r"^算法$"],
    SectionType.EXPERIMENT: [r"^3?\s*experiment", r"^实验$", r"^3?\s*实验"],
    SectionType.RESULTS: [r"^3?\s*result", r"^结果$", r"^3?\s*结果"],
    SectionType.EVALUATION: [r"^3?\s*evaluation", r"^评估$", r"^3?\s*评估"],
    SectionType.DISCUSSION: [r"^discussion$", r"^讨论$"],
    SectionType.CONCLUSION: [r"^conclusion$", r"^结论$", r"^5?\s*结论"],
    SectionType.REFERENCES: [r"^references$", r"^参考文献$", r"^reference"],
    SectionType.APPENDIX: [r"^appendix", r"^附录"],
}


class SectionDetector:
    """Detect and parse paper sections."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns."""
        self._patterns: dict[SectionType, list[re.Pattern]] = {}
        for section_type, patterns in SECTION_PATTERNS.items():
            self._patterns[section_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def detect_sections(self, pages: list, full_text: str) -> list[Section]:
        """Detect sections in the paper."""
        sections = []
        current_section: Optional[Section] = None

        # Split text into lines and analyze
        lines = full_text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line is a section header
            section_type = self._detect_section_type(line)
            if section_type != SectionType.UNKNOWN:
                # Save previous section
                if current_section:
                    current_section.end_page = self._estimate_page(i, len(lines))
                    sections.append(current_section)

                # Start new section
                current_section = Section(
                    section_type=section_type,
                    title=line,
                    content="",
                    start_page=self._estimate_page(i, len(lines)),
                    end_page=0,
                    level=self._detect_section_level(line),
                )
            elif current_section:
                # Add content to current section
                current_section.content += line + "\n"

        # Add last section
        if current_section:
            current_section.end_page = pages[-1].page_number if pages else 1
            sections.append(current_section)

        return sections

    def _detect_section_type(self, line: str) -> SectionType:
        """Detect section type from title."""
        # Must be relatively short to be a header
        if len(line) > 100:
            return SectionType.UNKNOWN

        for section_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.match(line):
                    return section_type

        return SectionType.UNKNOWN

    def _detect_section_level(self, line: str) -> int:
        """Detect section level (1 for main, 2 for subsection)."""
        # Check for numbered sections like "1 Introduction" vs "1.1 Methods"
        if re.match(r"^\d+\.\d+\s", line):
            return 2
        return 1

    def _estimate_page(self, line_index: int, total_lines: int) -> int:
        """Estimate page number from line index."""
        # Assume ~50 lines per page
        return (line_index // 50) + 1

    def get_section_by_type(self, sections: list[Section], section_type: SectionType) -> Optional[Section]:
        """Get a specific section by type."""
        for section in sections:
            if section.section_type == section_type:
                return section
        return None
