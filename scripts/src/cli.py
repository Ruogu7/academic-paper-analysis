"""CLI interface for paper-reader."""

import sys
from pathlib import Path

import click
from loguru import logger

from .agent import PaperAgent


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


@click.command()
@click.argument("paper_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the summary (default: ./output/{paper_name}_summary.md)",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="LLM model to use (default: gpt-4o)",
)
@click.option(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="LLM temperature (default: 0.7)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    paper_path: Path,
    output: Path | None,
    model: str | None,
    temperature: float,
    verbose: bool,
) -> None:
    """Read and summarize an academic paper.

    PAPER_PATH: Path to the PDF file to analyze.
    """
    setup_logging(verbose)

    logger.info(f"Paper Reader - Reading: {paper_path.name}")

    # Create output directory if not specified
    if output is None:
        output = Path("./output")
    output.mkdir(parents=True, exist_ok=True)

    # Output file path
    output_file = output / f"{paper_path.stem}_summary.md"

    try:
        # Initialize agent
        agent = PaperAgent(model=model, temperature=temperature)

        # Run analysis
        summary = agent.read_and_summarize(paper_path, output_file)

        # Print summary to console as well
        logger.info("=" * 60)
        logger.info(f"Summary generated: {summary.title}")
        logger.info(f"Output saved to: {output_file}")
        logger.info("=" * 60)

        # Print first part of summary
        from .output import MarkdownGenerator

        mg = MarkdownGenerator()
        content = mg.generate(summary)
        lines = content.split("\n")[:30]  # First 30 lines
        print("\nPreview:")
        print("\n".join(lines))
        print("\n... (full content saved to file)")

    except Exception as e:
        logger.error(f"Failed to process paper: {e}")
        if verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
