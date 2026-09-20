"""Multi-turn conversation handler for paper analysis."""

from dataclasses import dataclass, field
from typing import Optional

from .client import LLMClient, Message


@dataclass
class Turn:
    """Single conversation turn."""

    role: str
    content: str
    analysis_type: Optional[str] = None  # "introduction", "innovation", etc.


@dataclass
class Conversation:
    """Multi-turn conversation for paper analysis."""

    system_prompt: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, role: str, content: str, analysis_type: Optional[str] = None) -> None:
        """Add a turn to the conversation."""
        self.turns.append(Turn(role=role, content=content, analysis_type=analysis_type))

    def to_messages(self) -> list[Message]:
        """Convert to LLM message format."""
        messages = [Message(role="system", content=self.system_prompt)]
        for turn in self.turns:
            messages.append(Message(role=turn.role, content=turn.content))
        return messages


class MultiTurnAnalyzer:
    """Handle multi-turn conversations for deep paper analysis."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.conversation: Optional[Conversation] = None

    def start_analysis(
        self,
        paper_content: str,
        system_prompt: str,
    ) -> Conversation:
        """Start a new analysis conversation."""
        self.conversation = Conversation(system_prompt=system_prompt)

        # Add initial paper content
        self.conversation.add_turn(
            role="user",
            content=f"以下是论文的完整内容，请仔细阅读后再回答问题：\n\n{paper_content[:50000]}",
            analysis_type="initial",
        )

        return self.conversation

    def ask(self, question: str, analysis_type: str) -> str:
        """Ask a follow-up question in the conversation."""
        if not self.conversation:
            raise ValueError("Must start analysis first")

        self.conversation.add_turn(role="user", content=question, analysis_type=analysis_type)

        messages = self.conversation.to_messages()
        response = self.llm.complete(messages=messages[1:])  # Skip system prompt in messages

        self.conversation.add_turn(role="assistant", content=response.content)

        return response.content

    def extract_structured_info(
        self,
        paper_sections: dict,
        extraction_prompt: str,
    ) -> dict:
        """Extract structured information from paper sections."""
        # Build content from sections
        content_parts = []
        for section_name, section_content in paper_sections.items():
            if section_content:
                content_parts.append(f"## {section_name}\n{section_content}")

        combined_content = "\n\n".join(content_parts)

        prompt = f"""请从以下论文内容中提取结构化信息。

{extraction_prompt}

论文内容：
{combined_content}

请以JSON格式返回提取的信息。"""

        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.conversation.system_prompt if self.conversation else "",
        )

        return self._parse_json_response(response.content)

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        import json

        # Try to find JSON in response
        try:
            # First try direct parse
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON-like structure
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return {"raw_content": content}


class SequentialAnalyzer:
    """Analyze paper sections sequentially for comprehensive understanding."""

    def __init__(self, llm: LLMClient, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.findings: dict[str, str] = {}

    def analyze_introduction(self, content: str) -> dict:
        """Analyze introduction section."""
        prompt = f"""请分析以下论文引言部分，提取：
1. 研究背景
2. 研究问题
3. 研究目标
4. 主要贡献

内容：
{content}
"""
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.system_prompt,
        )
        self.findings["introduction"] = response.content
        return {"introduction": response.content}

    def analyze_method(self, content: str) -> dict:
        """Analyze method section."""
        prompt = f"""请分析以下论文的方法部分，提取：
1. 核心技术/模型/算法
2. 关键步骤和细节
3. 公式和架构
4. 方法的核心思想

内容：
{content}
"""
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.system_prompt,
        )
        self.findings["method"] = response.content
        return {"method": response.content}

    def analyze_results(self, content: str) -> dict:
        """Analyze results section."""
        prompt = f"""请分析以下论文的实验结果部分，提取：
1. 数据集信息
2. 具体数值结果
3. 与基线对比
4. 关键图表

内容：
{content}
"""
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.system_prompt,
        )
        self.findings["results"] = response.content
        return {"results": response.content}

    def analyze_conclusion(self, content: str) -> dict:
        """Analyze conclusion section."""
        prompt = f"""请分析以下论文的结论部分，提取：
1. 主要结论
2. 局限性
3. 未来工作

内容：
{content}
"""
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.system_prompt,
        )
        self.findings["conclusion"] = response.content
        return {"conclusion": response.content}

    def synthesize(self) -> str:
        """Synthesize all findings into a comprehensive summary."""
        prompt = f"""请根据以下各个部分的分析结果，生成一份完整的、结构化的论文总结。

{chr(10).join([f"## {k.upper()}\n{v}" for k, v in self.findings.items()])}

请按照以下格式组织：
1. 论文简介
2. 论文创新点
3. 核心技术
4. 实验结果
5. 结论与展望
"""
        response = self.llm.complete(
            messages=[Message(role="user", content=prompt)],
            system_prompt=self.system_prompt,
        )
        return response.content
