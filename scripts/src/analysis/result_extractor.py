"""Result extraction from academic papers."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentResult:
    """Represents an experimental result."""

    dataset: str
    metric: str
    value: str
    baseline_value: Optional[str] = None
    improvement: Optional[str] = None
    table_or_figure: Optional[str] = None


@dataclass
class ResultSection:
    """Represents a result section."""

    title: str
    content: str
    results: list[ExperimentResult] = field(default_factory=list)


@dataclass
class ExtractedResults:
    """Complete extracted results."""

    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    results: list[ExperimentResult] = field(default_factory=list)
    tables: dict[str, str] = field(default_factory=dict)  # table_id -> description
    figures: dict[str, str] = field(default_factory=dict)  # figure_id -> description


class ResultExtractor:
    """Extract experimental results from paper text."""

    # Common metric patterns
    METRIC_PATTERNS = [
        r"(accuracy|ACC|prec[^\s]+|recall|F1[- ]?score|BLEU|ROUGE|METEOR|CIDEr)",
        r"(AUC|ROC|FPR|TPR|Precision|Recall)",
        r"(PSNR|SSIM|LPIPS|FID|IS)",
        r"(loss|Loss|error|Error|latency|perplexity|PPL)",
    ]

    # Common table indicators
    TABLE_PATTERNS = [
        r"(?:Table|表|Tab\.)\s*(\d+[a-zA-Z]?)",
    ]

    # Common figure indicators
    FIGURE_PATTERNS = [
        r"(?:Figure|图|Fig\.)\s*(\d+[a-zA-Z]?)",
    ]

    # Dataset patterns
    DATASET_PATTERNS = [
        r"(?:on|dataset|data|benchmark|bench|test)\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s|$|,)",
    ]

    def __init__(self):
        self._metric_patterns = [re.compile(p, re.IGNORECASE) for p in self.METRIC_PATTERNS]
        self._table_patterns = [re.compile(p, re.IGNORECASE) for p in self.TABLE_PATTERNS]
        self._figure_patterns = [re.compile(p, re.IGNORECASE) for p in self.FIGURE_PATTERNS]
        self._dataset_patterns = [re.compile(p, re.IGNORECASE) for p in self.DATASET_PATTERNS]

    def extract_results(self, text: str) -> list[ExperimentResult]:
        """Extract all experimental results from text."""
        results = []

        # Split into potential result blocks
        lines = text.split("\n")
        current_dataset = "General"

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for dataset mention
            dataset = self._extract_dataset(line)
            if dataset:
                current_dataset = dataset
                continue

            # Check for metric-value pairs
            value_matches = self._extract_metric_values(line)
            for metric, value in value_matches:
                results.append(
                    ExperimentResult(
                        dataset=current_dataset,
                        metric=metric,
                        value=value,
                    )
                )

        return results

    def extract_comprehensive(self, text: str) -> ExtractedResults:
        """Extract comprehensive results including datasets, metrics, tables, figures."""
        extracted = ExtractedResults()

        # Extract tables
        for pattern in self._table_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                table_id = match.group(1)
                # Get context around table
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                extracted.tables[table_id] = context

        # Extract figures
        for pattern in self._figure_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                figure_id = match.group(1)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                extracted.figures[figure_id] = context

        # Extract datasets
        lines = text.split("\n")
        for line in lines:
            dataset = self._extract_dataset(line)
            if dataset and dataset not in extracted.datasets:
                extracted.datasets.append(dataset)

        # Extract metrics
        for line in lines:
            metrics = self._extract_metrics(line)
            for metric in metrics:
                if metric not in extracted.metrics:
                    extracted.metrics.append(metric)

        # Extract results
        extracted.results = self.extract_results(text)

        return extracted

    def _extract_dataset(self, line: str) -> Optional[str]:
        """Extract dataset name from line."""
        for pattern in self._dataset_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip()
        return None

    def _extract_metrics(self, line: str) -> list[str]:
        """Extract metric names from line."""
        metrics = []
        for pattern in self._metric_patterns:
            matches = pattern.finditer(line)
            for match in matches:
                metrics.append(match.group(1))
        return metrics

    def _extract_metric_values(self, line: str) -> list[tuple[str, str]]:
        """Extract metric-value pairs from a line."""
        matches = []

        # Pattern 1: metric: value or metric = value
        pattern1 = r"([A-Za-z][A-Za-z\s]{1,25}?(?:score|accuracy|rate|error|latency|throughput|perplexity|F1|BLEU|ROUGE|METEOR))\s*[:=]\s*(\d+(?:\.\d+)?%?)"

        for match in re.finditer(pattern1, line, re.IGNORECASE):
            metric = match.group(1).strip()
            value = match.group(2).strip()
            matches.append((metric, value))

        # Pattern 2: value% or value °/o
        pattern2 = r"(\d+(?:\.\d+)?)\s*%"

        return matches

    def extract_tables(self, text: str) -> list[dict]:
        """Extract table information."""
        tables = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            for pattern in self._table_patterns:
                match = pattern.search(line)
                if match:
                    table_id = match.group(1)
                    # Get surrounding context
                    context_lines = lines[max(0, i-1):min(len(lines), i+3)]
                    tables.append({
                        "id": table_id,
                        "header": line.strip(),
                        "context": "\n".join(context_lines),
                    })
                    break

        return tables
