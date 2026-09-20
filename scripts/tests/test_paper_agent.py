"""Tests for paper agent title/authors extraction."""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.paper_agent import PaperAgent


class TestTitleAuthorsExtraction:
    """Test LLM-based title and authors extraction."""

    @patch("src.agent.paper_agent.LLMClient")
    def test_extract_title_authors_with_llm_success(self, mock_llm_class):
        """Test successful title and authors extraction via LLM."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '''{
            "title": "Test Paper Title: A Comprehensive Study",
            "authors": "Zhang, San; Li, Si; Wang, Wu"
        }'''
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = PaperAgent()
        first_page_text = "Test Paper Title: A Comprehensive Study\nZhang, San\nLi, Si\nWang, Wu\nAbstract..."

        title, authors = agent._extract_title_authors_with_llm(
            first_page_text, "Old Title", "Fallback"
        )

        assert title == "Test Paper Title: A Comprehensive Study"
        assert "Zhang, San" in authors
        assert "Li, Si" in authors

    @patch("src.agent.paper_agent.LLMClient")
    def test_extract_title_authors_with_llm_json_in_codeblock(self, mock_llm_class):
        """Test LLM returns JSON in markdown code block."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # LLM returns JSON in code block
        mock_response.content = '''```json
{
    "title": "Another Paper Title",
    "authors": "John Doe, Jane Smith"
}
```'''
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = PaperAgent()
        first_page_text = "Another Paper Title\nJohn Doe\nJane Smith"

        title, authors = agent._extract_title_authors_with_llm(
            first_page_text, "Old Title", "Fallback"
        )

        assert title == "Another Paper Title"
        assert "John Doe" in authors

    @patch("src.agent.paper_agent.LLMClient")
    def test_extract_title_authors_fallback_on_llm_error(self, mock_llm_class):
        """Test fallback to regex on LLM failure."""
        mock_llm = MagicMock()
        # Simulate LLM API error
        mock_llm.complete.side_effect = Exception("API Error")
        mock_llm_class.return_value = mock_llm

        agent = PaperAgent()
        first_page_text = "Valid Title\nUniversity of Test\nAbstract..."

        title, authors = agent._extract_title_authors_with_llm(
            first_page_text, "Old Title", "Fallback"
        )

        # Should fallback to regex method - title should be one of the valid options
        assert title in ["Valid Title", "Old Title", "Fallback"]

    @patch("src.agent.paper_agent.LLMClient")
    def test_extract_title_authors_fallback_on_invalid_json(self, mock_llm_class):
        """Test fallback on invalid JSON response."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # Return invalid JSON
        mock_response.content = "This is not valid JSON"
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = PaperAgent()
        first_page_text = "Some Title\nAuthor Name"

        title, authors = agent._extract_title_authors_with_llm(
            first_page_text, "Old Title", "Fallback"
        )

        # Should fallback to regex method
        assert title in ["Some Title", "Old Title", "Fallback"]

    @patch("src.agent.paper_agent.LLMClient")
    def test_extract_title_authors_empty_response(self, mock_llm_class):
        """Test handling of empty JSON fields."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"title": "", "authors": ""}'
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        agent = PaperAgent()
        first_page_text = "Original Title\nOriginal Author"

        title, authors = agent._extract_title_authors_with_llm(
            first_page_text, "Current Title", "Default"
        )

        # Should use fallback values when LLM returns empty
        assert title == "Current Title"
        assert authors == "Unknown"
