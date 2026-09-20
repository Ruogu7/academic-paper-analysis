"""GitHub integration for fetching project code and documentation."""

import base64
import re
from typing import Optional

import requests
from loguru import logger


class GitHubIntegration:
    """Handle GitHub repository fetching for paper context."""

    def __init__(self, max_repos: int = 2, max_files: int = 3):
        """Initialize GitHub integration.

        Args:
            max_repos: Maximum number of repositories to fetch from
            max_files: Maximum number of source files to fetch per repo
        """
        self.max_repos = max_repos
        self.max_files = max_files

    def extract_github_urls(self, text: str) -> list[str]:
        """Extract GitHub repository URLs from text.

        Args:
            text: Text content to search for GitHub URLs

        Returns:
            List of GitHub repository URLs
        """
        # Patterns for GitHub URLs
        patterns = [
            r'https?://github\.com/[\w\-]+/[\w\-.]+',
            r'github\.com/[\w\-]+/[\w\-.]+',
            r'git@github\.com:[\w\-]+/[\w\-.]+\.git',
        ]

        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            urls.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            # Clean up URL (remove trailing .git, etc.)
            clean_url = url.rstrip('/').rstrip('.git')
            if clean_url not in seen:
                seen.add(clean_url)
                unique_urls.append(clean_url)

        return unique_urls

    def fetch_readme(self, owner: str, repo: str) -> str:
        """Fetch README content from GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            README content as string, empty string if not found
        """
        try:
            # Try to get README with common names
            readme_names = ["README.md", "README.rst", "README.txt", "README"]

            for name in readme_names:
                url = f"https://api.github.com/repos/{owner}/{repo}/contents/{name}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if "content" in data:
                        # Decode base64 content
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        logger.info(f"Fetched README ({name}) from {owner}/{repo}")
                        return content[:5000]  # Limit to 5000 chars

            logger.info(f"No README found for {owner}/{repo}")
            return ""

        except Exception as e:
            logger.warning(f"Failed to fetch README from {owner}/{repo}: {e}")
            return ""

    def fetch_file(self, owner: str, repo: str, file_path: str) -> str:
        """Fetch a specific file from GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path to the file

        Returns:
            File content as string, empty string if not found
        """
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    logger.info(f"Fetched file {file_path} from {owner}/{repo}")
                    return content[:3000]  # Limit to 3000 chars

            return ""

        except Exception as e:
            logger.warning(f"Failed to fetch file {file_path} from {owner}/{repo}: {e}")
            return ""

    def list_files(self, owner: str, repo: str) -> list[str]:
        """List files in repository root.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List of file paths in repository
        """
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                files = response.json()
                return [f["path"] for f in files if f.get("type") == "file"]

            return []

        except Exception as e:
            logger.warning(f"Failed to list files from {owner}/{repo}: {e}")
            return []

    def extract_key_code_files(self, file_list: list[str]) -> list[str]:
        """Extract key code files from repository file list.

        Args:
            file_list: List of file paths in repository

        Returns:
            List of key code files to include in summary
        """
        # Priority patterns for important files
        priority_patterns = [
            r'\.py$',
            r'\.js$',
            r'\.ts$',
            r'\.cpp$',
            r'\.c$',
            r'\.java$',
            r'model\.',
            r'train\.',
            r'engine\.',
            r'core\.',
            r'config\.',
        ]

        key_files = []

        for file_path in file_list:
            # Skip test files, docs, etc.
            if any(x in file_path.lower() for x in ["test", "doc", "example", "demo"]):
                continue
            if not any(re.search(p, file_path) for p in priority_patterns):
                continue
            key_files.append(file_path)

        return key_files[:self.max_files]  # Limit to max_files

    def fetch_project_context(self, github_urls: list[str]) -> str:
        """Fetch project context from GitHub repositories.

        Args:
            github_urls: List of GitHub repository URLs

        Returns:
            Formatted project context string
        """
        if not github_urls:
            return ""

        context_parts = []

        for url in github_urls[:self.max_repos]:
            # Extract owner and repo from URL
            match = re.search(r'github\.com/([\w\-]+)/([\w\-.]+)', url)
            if not match:
                continue

            owner, repo = match.groups()
            repo = repo.rstrip('/').rstrip('.git')

            # Get README
            readme = self.fetch_readme(owner, repo)
            if readme:
                context_parts.append(f"### 仓库: {owner}/{repo}\n\n**README:**\n{readme[:2000]}")

            # Get key source files
            files = self.list_files(owner, repo)
            key_files = self.extract_key_code_files(files)

            for file_path in key_files:
                content = self.fetch_file(owner, repo, file_path)
                if content:
                    context_parts.append(f"\n**文件: {file_path}**\n```\n{content[:800]}\n```")

        if context_parts:
            return "\n\n".join(context_parts)
        return ""


def build_prompt_with_context(
    title: str,
    authors: str,
    content: str,
    project_context: str,
) -> str:
    """Build prompt with project context for LLM.

    Args:
        title: Paper title
        authors: Paper authors
        content: Paper content
        project_context: Project file context

    Returns:
        Formatted prompt string
    """
    return f"""## 论文信息
标题: {title}
作者: {authors}

## 论文内容
{content}

## 项目代码上下文
{project_context}

请根据以上内容，生成完整的论文总结。

**重要要求**：在"核心技术"或"方法细节"部分，你必须主动引用上面的项目代码来解释论文中的核心算法和实现细节。
1. 保持专业的学术语言
2. 代码示例应简洁明了，每段不超过5行
3. 使用```代码块格式，并在代码后简要解释其含义
4. 只引用最关键的代码片段，不要过度引用"""
