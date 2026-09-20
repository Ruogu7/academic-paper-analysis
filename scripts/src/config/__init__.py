"""Configuration management for paper-reader."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models for different providers
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4o"


def _detect_claude_code_model() -> tuple[Optional[str], Optional[str]]:
    """Detect Claude Code configured model and API key.

    Returns:
        tuple of (model, api_key) if detected, (None, None) otherwise
    """
    # Priority: 1. Explicit model config 2. API key detection

    # Check for explicit model configuration first
    explicit_model = os.environ.get("DEFAULT_MODEL")
    if explicit_model:
        # Check for corresponding API key
        if "claude" in explicit_model.lower():
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY")
            if key:
                return (explicit_model, key)
        elif "gpt" in explicit_model.lower() or explicit_model.startswith("o1"):
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                return (explicit_model, key)

    # Detect based on API keys
    # Anthropic API
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY")
    if anthropic_key:
        # Check if user specified a model, otherwise use default
        model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return (model, anthropic_key)

    # OpenAI API
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return (model, openai_key)

    # Check for other common variables
    for var in ["CLAUDE_API_KEY", "API_KEY"]:
        key = os.environ.get(var)
        if key:
            # Try to determine provider from key format or default to Anthropic
            return (DEFAULT_ANTHROPIC_MODEL, key)

    return (None, None)


# Detect Claude Code configuration
_detected_model, _detected_api_key = _detect_claude_code_model()


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    # Priority: 1. Claude Code detected 2. Environment variable 3. Default
    openai_api_key: Optional[str] = Field(
        default=_detected_api_key if _detected_model and "gpt" in _detected_model else None,
        alias="OPENAI_API_KEY"
    )
    anthropic_api_key: Optional[str] = Field(
        default=_detected_api_key if _detected_model and "claude" in _detected_model else None,
        alias="ANTHROPIC_API_KEY"
    )
    minimax_api_key: Optional[str] = Field(default=None, alias="MINIMAX_API_KEY")
    minimax_group_id: Optional[str] = Field(default=None, alias="MINIMAX_GROUP_ID")

    # Default model: prefer Claude Code detected, then MiniMax
    default_model: str = Field(
        default=_detected_model or "minimax/MiniMax-M2.5",
        alias="DEFAULT_MODEL"
    )
    vision_model: str = Field(
        default=_detected_model or "minimax/MiniMax-M2.5",
        alias="VISION_MODEL"
    )

    # Output Settings
    output_dir: Path = Field(default=Path.home() / "paper_reader_output", alias="OUTPUT_DIR")
    include_figures: bool = Field(default=True, alias="INCLUDE_FIGURES")

    # Processing Settings
    max_pages_per_chunk: int = Field(default=10, alias="MAX_PAGES_PER_CHUNK")
    concurrency: int = Field(default=4, alias="CONCURRENCY")

    @property
    def is_claude_code_mode(self) -> bool:
        """Check if using Claude Code configured model."""
        return "claude" in self.default_model.lower() or "gpt" in self.default_model.lower()


settings = Settings()
