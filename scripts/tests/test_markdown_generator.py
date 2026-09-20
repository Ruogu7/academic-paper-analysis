"""Tests for markdown generator format validation."""

import pytest
from unittest.mock import patch, MagicMock

from src.output.markdown_generator import MarkdownGenerator, PaperSummary


class TestMarkdownFormatValidation:
    """Test Markdown format validation."""

    @patch("src.output.markdown_generator.LLMClient")
    def test_validate_and_fix_format_removes_section_numbers(self, mock_llm_class):
        """Test that section numbers are removed by LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # LLM should return fixed content without section numbers
        mock_response.content = """# Valid Title

## 摘要

content

## 研究背景

content
"""
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        input_md = """# Title

### 2.1 研究背景

content
"""

        result = gen._validate_and_fix_format(input_md)

        # Verify LLM was called
        mock_llm.complete.assert_called_once()
        # Result should be the fixed content from LLM
        assert "2.1" not in result

    @patch("src.output.markdown_generator.LLMClient")
    def test_validate_and_fix_format_adds_table_separators(self, mock_llm_class):
        """Test that table separators are added by LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """# Title

| Col1 | Col2 |
|------|------|
| Val1 | Val2 |
"""
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        input_md = """# Title

| Col1 | Col2
| Val1 | Val2
"""

        result = gen._validate_and_fix_format(input_md)

        mock_llm.complete.assert_called_once()
        # Check for proper table separator (|------| or |---|)
        assert "|------|" in result or "|---|" in result

    @patch("src.output.markdown_generator.LLMClient")
    def test_validate_and_fix_format_removes_standalone_dashes(self, mock_llm_class):
        """Test that standalone --- separators are removed."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """# Title

## 摘要

content

## 正文

content
"""
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        input_md = """# Title

---

## 摘要

content

---

## 正文

content
"""

        result = gen._validate_and_fix_format(input_md)

        mock_llm.complete.assert_called_once()

    @patch("src.output.markdown_generator.LLMClient")
    def test_validate_and_fix_format_fallback_on_error(self, mock_llm_class):
        """Test fallback to original content on LLM error."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("API Error")
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        input_md = "# Original Title\n\n## 章节"

        result = gen._validate_and_fix_format(input_md)

        # Should return original content on error
        assert result == input_md

    @patch("src.output.markdown_generator.LLMClient")
    def test_validate_and_fix_format_fallback_on_invalid_response(self, mock_llm_class):
        """Test fallback when LLM returns invalid content."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        # Return too short content (less than 50% of original)
        mock_response.content = "Short"
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        input_md = "# " + "x" * 1000 + "\n\n## 章节"

        result = gen._validate_and_fix_format(input_md)

        # Should return original content when LLM returns invalid content
        assert result == input_md


class TestMarkdownGeneratorSave:
    """Test MarkdownGenerator save method with validation."""

    @patch("src.output.markdown_generator.LLMClient")
    def test_save_calls_format_validation(self, mock_llm_class):
        """Test that save method calls format validation."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "# Fixed Title\n\n## 摘要"
        mock_llm.complete.return_value = mock_response
        mock_llm_class.return_value = mock_llm

        gen = MarkdownGenerator()
        summary = PaperSummary(
            title="Test Title",
            authors="Test Author"
        )

        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"
            gen.save(summary, output_path)

            # Verify LLM was called for format validation
            mock_llm.complete.assert_called_once()

            # Verify file was written
            assert output_path.exists()


class TestMarkdownSpacing:
    """Test Markdown spacing fixes for better rendering."""

    def test_add_spacing_around_bold_markers(self):
        """Test that bold markers (**) have spaces on both sides."""
        gen = MarkdownGenerator()

        # Input without spaces around bold markers
        input_text = "这是**粗体**文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        # Expected: spaces around ** markers
        assert "这是 **粗体** 文本" == result

    def test_add_spacing_around_italic_markers(self):
        """Test that italic markers (*) have spaces on both sides."""
        gen = MarkdownGenerator()

        # Input without spaces around italic markers
        input_text = "这是*斜体*文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        # Expected: spaces around * markers
        assert "这是 *斜体* 文本" == result

    def test_add_spacing_around_code_markers(self):
        """Test that code markers (`) have spaces on both sides."""
        gen = MarkdownGenerator()

        # Input without spaces around code markers
        input_text = "这是`代码`文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        # Expected: spaces around ` markers
        assert "这是 `代码` 文本" == result

    def test_add_spacing_around_mixed_markers(self):
        """Test mixed bold and italic markers."""
        gen = MarkdownGenerator()

        input_text = "这是**粗体**和*斜体*文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        assert "这是 **粗体** 和 *斜体* 文本" == result

    def test_add_spacing_around_bold_italic_markers(self):
        """Test bold italic markers (***)."""
        gen = MarkdownGenerator()

        input_text = "这是***粗体斜体***文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        assert "这是 ***粗体斜体*** 文本" == result

    def test_add_spacing_around_strikethrough_markers(self):
        """Test strikethrough markers (~~)."""
        gen = MarkdownGenerator()

        input_text = "这是~~删除线~~文本"
        result = gen._add_spacing_around_markdown_symbols(input_text)

        assert "这是 ~~删除线~~ 文本" == result


class TestChineseEnglishSpacing:
    """Test Chinese-English spacing for better readability."""

    def test_add_space_between_chinese_and_english(self):
        """Test that space is added between Chinese and English text."""
        gen = MarkdownGenerator()

        input_text = "这是Python代码"
        result = gen._add_spacing_between_chinese_english(input_text)

        assert "这是 Python 代码" == result

    def test_add_space_english_to_chinese(self):
        """Test space from English to Chinese."""
        gen = MarkdownGenerator()

        input_text = "Hello世界"
        result = gen._add_spacing_between_chinese_english(input_text)

        assert "Hello 世界" == result

    def test_add_space_in_mixed_text(self):
        """Test space in mixed Chinese-English text."""
        gen = MarkdownGenerator()

        input_text = "深度学习DeepLearning是机器学习的分支"
        result = gen._add_spacing_between_chinese_english(input_text)

        # Should add space between Chinese and English
        assert "深度学习 DeepLearning 是机器学习的分支" == result

    def test_add_space_before_parentheses(self):
        """Test that space is added before English parentheses."""
        gen = MarkdownGenerator()

        input_text = "这是function(x)调用"
        result = gen._add_spacing_between_chinese_english(input_text)

        # Should add space between Chinese and function, but keep parentheses attached to function
        assert "这是 function(x) 调用" == result

    def test_add_space_after_punctuation(self):
        """Test that space is added after punctuation before English."""
        gen = MarkdownGenerator()

        input_text = "大家好,hello world"
        result = gen._add_spacing_between_chinese_english(input_text)

        assert "大家好, hello world" == result
