#!/usr/bin/env python3
"""Paper Reader - CLI wrapper for Claude Code integration."""

import argparse
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agent import PaperAgent
from src.config import settings


def run_config_wizard():
    """Run the model configuration wizard."""
    from src.config.model_wizard import run_wizard
    run_wizard()


def main():
    parser = argparse.ArgumentParser(
        description="Paper Reader - Analyze academic papers with AI"
    )
    parser.add_argument(
        "paper_path",
        type=str,
        nargs="?",  # Make it optional for --config
        help="Path to the PDF paper file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for summary (default: ./output/{paper_name}_summary.md)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="LLM model to use",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key from environment",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Run model configuration wizard",
    )

    args = parser.parse_args()

    # Handle config wizard
    if args.config:
        run_config_wizard()
        return

    # Handle API key
    if args.api_key:
        os.environ["MINIMAX_API_KEY"] = args.api_key

    # Validate paper path
    paper_path = Path(args.paper_path)
    if not paper_path.exists():
        print(f"Error: Paper file not found: {paper_path}")
        sys.exit(1)

    # Handle output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{paper_path.stem}_summary.md"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize agent
    model = args.model or settings.default_model or "minimax/MiniMax-M2.5"

    if args.verbose:
        print(f"Using model: {model}")
        print(f"Paper: {paper_path}")
        print(f"Output: {output_path}")

    agent = PaperAgent(model=model, temperature=args.temperature)

    # Run analysis
    try:
        summary = agent.read_and_summarize(paper_path, output_path)

        print("\n" + "=" * 60)
        print(f"Analysis Complete!")
        print(f"Title: {summary.title}")
        print(f"Output saved to: {output_path}")
        print("=" * 60)

        # Print preview
        from src.output import MarkdownGenerator
        mg = MarkdownGenerator()
        content = mg.generate(summary)
        lines = content.split("\n")[:20]

        print("\nPreview:")
        print("\n".join(lines))
        print("\n... (full content in file)")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
