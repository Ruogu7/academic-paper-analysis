"""Figure analyzer using OCR and LLM."""

import io
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    import pytesseract
except ImportError:
    pytesseract = None

from PIL import Image

from ..llm import LLMClient, Message
from ..prompts import SYSTEM_PROMPT


@dataclass
class FigureAnalysis:
    """Analysis result for a figure."""

    figure_id: str
    description: str
    ocr_text: str = ""
    key_findings: list[str] = None

    def __post_init__(self):
        if self.key_findings is None:
            self.key_findings = []


class FigureAnalyzer:
    """Analyze figures using OCR and LLM."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

        if pytesseract is None:
            logger.warning("pytesseract not installed, OCR will be skipped")

    def analyze_figure(
        self,
        image: Image.Image,
        figure_id: str,
        caption: str = "",
    ) -> FigureAnalysis:
        """Analyze a figure using OCR and LLM."""
        # First, try OCR
        ocr_text = ""
        if pytesseract:
            try:
                ocr_text = pytesseract.image_to_string(image)
                logger.debug(f"OCR extracted {len(ocr_text)} characters")
            except Exception as e:
                logger.warning(f"OCR failed for {figure_id}: {e}")

        # Build prompt for LLM
        prompt = self._build_analysis_prompt(image, caption, ocr_text)

        # Call LLM
        try:
            response = self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                system_prompt=SYSTEM_PROMPT,
            )

            return FigureAnalysis(
                figure_id=figure_id,
                description=response.content,
                ocr_text=ocr_text,
            )
        except Exception as e:
            logger.error(f"LLM analysis failed for {figure_id}: {e}")
            return FigureAnalysis(
                figure_id=figure_id,
                description=f"Figure {figure_id}" + (f": {caption}" if caption else ""),
                ocr_text=ocr_text,
            )

    def analyze_figures_batch(
        self,
        images: list[tuple[Image.Image, str, str]],
    ) -> list[FigureAnalysis]:
        """Analyze multiple figures in batch."""
        analyses = []

        for image, figure_id, caption in images:
            analysis = self.analyze_figure(image, figure_id, caption)
            analyses.append(analysis)

        return analyses

    def _build_analysis_prompt(
        self,
        image: Image.Image,
        caption: str,
        ocr_text: str,
    ) -> str:
        """Build analysis prompt for LLM."""
        prompt = """请分析以下图片的内容，并提供详细的描述。

"""

        if caption:
            prompt += f"图片标题/Caption: {caption}\n\n"

        if ocr_text:
            prompt += f"图片中的文字识别结果 (OCR):\n{ocr_text}\n\n"

        prompt += """请提供：
1. 图片的详细描述
2. 图片中的关键元素和信息
3. 图片在论文中的作用和意义

请用中文回答。"""

        return prompt


class SimpleFigureAnalyzer:
    """Simple figure analyzer without OCR (for when pytesseract is not available)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze_figure(
        self,
        image_bytes: bytes,
        figure_id: str,
        caption: str = "",
    ) -> FigureAnalysis:
        """Analyze a figure using LLM with vision capabilities."""
        # For now, just use caption
        prompt = f"""请根据以下信息描述这个图片：

图片编号: {figure_id}
图片标题: {caption}

请提供图片的详细描述，包括：
1. 图片的主要内容
2. 图表中的关键数据或关系
3. 图片传达的主要信息

请用中文回答，描述要详细准确。"""

        try:
            # Check if model supports vision
            response = self.llm.complete(
                messages=[Message(role="user", content=prompt)],
                system_prompt=SYSTEM_PROMPT,
            )

            return FigureAnalysis(
                figure_id=figure_id,
                description=response.content,
                ocr_text="",
            )
        except Exception as e:
            logger.error(f"Figure analysis failed: {e}")
            return FigureAnalysis(
                figure_id=figure_id,
                description=caption or f"Figure {figure_id}",
                ocr_text="",
            )
