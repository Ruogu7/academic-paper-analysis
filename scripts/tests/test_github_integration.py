"""Tests for GitHub project integration."""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.github_integration import GitHubIntegration, build_prompt_with_context


class TestGitHubLinkExtraction:
    """Test extracting GitHub links from paper content."""

    def test_extract_github_url_from_text(self):
        """Test extracting GitHub URL from paper text."""
        github = GitHubIntegration()

        text = """
        We implement our method using PyTorch and release the code at
        https://github.com/username/project-name
        """

        urls = github.extract_github_urls(text)

        # Should find at least one URL (may include both full and short formats)
        assert len(urls) >= 1
        has_url = any("github.com/username/project-name" in url for url in urls)
        assert has_url

    def test_extract_multiple_github_urls(self):
        """Test extracting multiple GitHub URLs."""
        github = GitHubIntegration()

        text = """
        Our code is available at https://github.com/author/repo1
        and we also provide a fork at https://github.com/author/repo2
        """

        urls = github.extract_github_urls(text)

        # Should find at least 2 unique URLs
        assert len(urls) >= 2

    def test_extract_github_url_with_different_formats(self):
        """Test extracting GitHub URLs in different formats."""
        github = GitHubIntegration()

        text = """
        Code: github.com/author/project
        URL: https://github.com/author/project
        SSH: git@github.com:author/project.git
        """

        urls = github.extract_github_urls(text)

        assert len(urls) >= 1

    def test_no_github_url_returns_empty(self):
        """Test that no GitHub URL returns empty list."""
        github = GitHubIntegration()

        text = """
        This paper describes a novel method for machine learning.
        """

        urls = github.extract_github_urls(text)

        assert urls == []


class TestGitHubRepoContent:
    """Test fetching GitHub repository content."""

    @patch("src.agent.github_integration.requests")
    def test_fetch_readme_content(self, mock_requests):
        """Test fetching README content from GitHub."""
        github = GitHubIntegration()

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "IyBSZWFkbWUgRmlsZQo9PT09PT09CgpUaGlzIGlzIHRoZSBwcm9qZWN0LiA="  # base64 of "# README File\n==========\n\nThis is the project. "
        }
        mock_requests.get.return_value = mock_response

        content = github.fetch_readme("username", "project")

        assert "README" in content or "project" in content

    @patch("src.agent.github_integration.requests")
    def test_fetch_repo_file_content(self, mock_requests):
        """Test fetching specific file from GitHub."""
        github = GitHubIntegration()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "cHJpbnQoImhlbGxvIik="  # base64 of 'print("hello")'
        }
        mock_requests.get.return_value = mock_response

        content = github.fetch_file("username", "project", "main.py")

        assert "hello" in content

    @patch("src.agent.github_integration.requests")
    def test_fetch_nonexistent_repo_returns_empty(self, mock_requests):
        """Test that fetching nonexistent repo returns empty."""
        github = GitHubIntegration()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests.get.return_value = mock_response

        content = github.fetch_readme("username", "nonexistent")

        assert content == ""


class TestProjectIntegration:
    """Test integrating project files into paper summary."""

    def test_generate_summary_with_project_context(self):
        """Test generating summary with project context."""
        project_context = """
        ## 项目代码

        核心算法实现:

        ```python
        def forward(x):
            return self.layer(x)
        ```
        """

        prompt = build_prompt_with_context(
            title="Test Paper",
            authors="Test Author",
            content="Paper content here",
            project_context=project_context
        )

        # Should include project context in prompt
        assert "项目代码" in prompt or "project" in prompt.lower()
        assert "forward" in prompt

    def test_extract_key_code_files(self):
        """Test extracting key code files from repo."""
        github = GitHubIntegration()

        file_list = [
            "src/model.py",
            "src/train.py",
            "src/utils.py",
            "tests/test_model.py",
            "README.md",
            "setup.py",
        ]

        key_files = github.extract_key_code_files(file_list)

        # Should prioritize source files
        assert "src/model.py" in key_files or "src/train.py" in key_files
