"""LLM client wrapper using litellm."""

from dataclasses import dataclass
from typing import Literal, Optional

from loguru import logger

try:
    import litellm
    from litellm import acompletion, completion
except ImportError:
    litellm = None

from ..config import settings


@dataclass
class Message:
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Response:
    """LLM response."""

    content: str
    model: str
    usage: Optional[dict] = None


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        if litellm is None:
            raise ImportError("litellm is required. Install with: pip install litellm")

        self.model = model or settings.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Setup MiniMax if using minimax model
        if self.model.startswith("minimax"):
            if settings.minimax_api_key:
                # litellm uses "minimax/<model_name>" format
                # Map to proper model name
                if "/" not in self.model:
                    self.model = "minimax/MiniMax-M2.5"
                # Set API key and group_id
                import os
                os.environ["MINIMAX_API_KEY"] = settings.minimax_api_key
                if settings.minimax_group_id:
                    os.environ["MINIMAX_GROUP_ID"] = settings.minimax_group_id
            else:
                raise ValueError("MiniMax API key is required when using minimax model")

        # Setup Anthropic/Claude model (from Claude Code or env)
        if "claude" in self.model.lower() and settings.anthropic_api_key:
            import os
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
            # Use anthropic format for litellm
            self.model = f"anthropic/{self.model}"

        # Setup OpenAI model (from Claude Code or env)
        if "gpt" in self.model.lower() and settings.openai_api_key:
            import os
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        # Setup API keys
        if settings.openai_api_key or settings.anthropic_api_key:
            litellm.drop_params = True

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> Response:
        """Synchronous completion."""
        # Build message list
        full_messages = []
        if system_prompt:
            full_messages.append(Message(role="system", content=system_prompt))
        full_messages.extend(messages)

        # Convert to dict format
        msg_dicts = [{"role": m.role, "content": m.content} for m in full_messages]

        logger.debug(f"Calling LLM with model: {self.model}")

        try:
            response = completion(
                model=self.model,
                messages=msg_dicts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            return Response(
                content=content,
                model=response.model,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def acomplete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> Response:
        """Asynchronous completion."""
        # Build message list
        full_messages = []
        if system_prompt:
            full_messages.append(Message(role="system", content=system_prompt))
        full_messages.extend(messages)

        # Convert to dict format
        msg_dicts = [{"role": m.role, "content": m.content} for m in full_messages]

        logger.debug(f"Calling LLM (async) with model: {self.model}")

        try:
            response = await acompletion(
                model=self.model,
                messages=msg_dicts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content or ""
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            return Response(
                content=content,
                model=response.model,
                usage=usage,
            )
        except Exception as e:
            logger.error(f"LLM async completion failed: {e}")
            raise


def create_client(
    model: Optional[str] = None,
    temperature: float = 0.7,
) -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient(model=model, temperature=temperature)
