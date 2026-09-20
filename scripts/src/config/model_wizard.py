"""Model configuration wizard for paper-reader.

Provides an interactive wizard for configuring LLM models from various providers.
Supports: Anthropic, OpenAI, DeepSeek, MiniMax, Qwen, Llama, Gemini, GLM, Kimi, and local models.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class ModelProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    MINIMAX = "minimax"
    QWEN = "qwen"
    LLAMA = "llama"
    GEMINI = "gemini"
    GLM = "glm"
    KIMI = "kimi"
    LOCAL = "local"


@dataclass
class ModelInfo:
    """Information about a cloud-based LLM model."""

    name: str
    provider: ModelProvider
    model_id: str
    requires_api_key: bool = True
    description: str = ""
    context_length: int = 0
    supports_vision: bool = False


@dataclass
class LocalModelInfo:
    """Information about a local/Ollama model."""

    name: str
    backend: str  # "ollama", "lm-studio", "llama.cpp", etc.
    model_id: str
    requires_gpu: bool = True
    description: str = ""
    memory_requirement_gb: int = 0


# Model catalog
ANTHROPIC_MODELS = [
    ModelInfo(
        name="Claude Sonnet 4.5",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        description="Balanced model for most tasks",
        context_length=200000,
        supports_vision=True,
    ),
    ModelInfo(
        name="Claude Opus 4.5",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-5-20250514",
        description="Most capable model for complex reasoning",
        context_length=200000,
        supports_vision=True,
    ),
    ModelInfo(
        name="Claude Haiku 4.5",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20250514",
        description="Fast and efficient model",
        context_length=200000,
        supports_vision=True,
    ),
    ModelInfo(
        name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-sonnet-20241022",
        description="Previous generation balanced model",
        context_length=200000,
        supports_vision=True,
    ),
]

OPENAI_MODELS = [
    ModelInfo(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        description="OpenAI's flagship model",
        context_length=128000,
        supports_vision=True,
    ),
    ModelInfo(
        name="GPT-4o-mini",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        description="Fast and cost-effective model",
        context_length=128000,
        supports_vision=True,
    ),
    ModelInfo(
        name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4-turbo",
        description="Previous generation flagship",
        context_length=128000,
        supports_vision=True,
    ),
    ModelInfo(
        name="GPT-4",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4",
        description="Reliable GPT-4 model",
        context_length=8192,
    ),
    ModelInfo(
        name="o1-preview",
        provider=ModelProvider.OPENAI,
        model_id="o1-preview",
        description="OpenAI's reasoning model",
        context_length=128000,
    ),
    ModelInfo(
        name="o1-mini",
        provider=ModelProvider.OPENAI,
        model_id="o1-mini",
        description="Fast reasoning model",
        context_length=128000,
    ),
]

DEEPSEEK_MODELS = [
    ModelInfo(
        name="DeepSeek V3",
        provider=ModelProvider.DEEPSEEK,
        model_id="deepseek/deepseek-chat",
        description="DeepSeek's latest model",
        context_length=64000,
    ),
    ModelInfo(
        name="DeepSeek Coder V2",
        provider=ModelProvider.DEEPSEEK,
        model_id="deepseek/deepseek-coder-v2",
        description="Specialized for code tasks",
        context_length=64000,
    ),
]

MINIMAX_MODELS = [
    ModelInfo(
        name="MiniMax M2.5",
        provider=ModelProvider.MINIMAX,
        model_id="minimax/MiniMax-M2.5",
        description="Default model for this project",
        context_length=32000,
    ),
    ModelInfo(
        name="MiniMax M2",
        provider=ModelProvider.MINIMAX,
        model_id="minimax/MiniMax-M2",
        description="Previous generation model",
        context_length=32000,
    ),
]

QWEN_MODELS = [
    ModelInfo(
        name="Qwen 2.5 72B",
        provider=ModelProvider.QWEN,
        model_id="qwen/qwen2.5-72b-instruct",
        description="Alibaba's large model",
        context_length=32768,
    ),
    ModelInfo(
        name="Qwen 2.5 7B",
        provider=ModelProvider.QWEN,
        model_id="qwen/qwen2.5-7b-instruct",
        description="Alibaba's efficient model",
        context_length=32768,
    ),
    ModelInfo(
        name="Qwen Coder 2.5",
        provider=ModelProvider.QWEN,
        model_id="qwen/qwen2.5-coder-7b-instruct",
        description="Specialized for code tasks",
        context_length=32768,
    ),
]

LLAMA_MODELS = [
    ModelInfo(
        name="Llama 3.1 405B",
        provider=ModelProvider.LLAMA,
        model_id="meta-llama/llama-3.1-405b-instruct",
        description="Meta's most capable model",
        context_length=128000,
    ),
    ModelInfo(
        name="Llama 3.1 70B",
        provider=ModelProvider.LLAMA,
        model_id="meta-llama/llama-3.1-70b-instruct",
        description="Meta's balanced model",
        context_length=128000,
    ),
    ModelInfo(
        name="Llama 3.1 8B",
        provider=ModelProvider.LLAMA,
        model_id="meta-llama/llama-3.1-8b-instruct",
        description="Meta's efficient model",
        context_length=128000,
    ),
    ModelInfo(
        name="Llama 3 70B",
        provider=ModelProvider.LLAMA,
        model_id="meta-llama/llama-3-70b-instruct",
        description="Previous generation flagship",
        context_length=8192,
    ),
]

GEMINI_MODELS = [
    ModelInfo(
        name="Gemini 1.5 Pro",
        provider=ModelProvider.GEMINI,
        model_id="gemini/gemini-1.5-pro",
        description="Google's flagship model",
        context_length=2000000,
        supports_vision=True,
    ),
    ModelInfo(
        name="Gemini 1.5 Flash",
        provider=ModelProvider.GEMINI,
        model_id="gemini/gemini-1.5-flash",
        description="Google's fast model",
        context_length=1000000,
        supports_vision=True,
    ),
    ModelInfo(
        name="Gemini 1.5 Flash-8B",
        provider=ModelProvider.GEMINI,
        model_id="gemini/gemini-1.5-flash-8b",
        description="Google's efficient model",
        context_length=1000000,
        supports_vision=True,
    ),
]

GLM_MODELS = [
    ModelInfo(
        name="GLM-4 Plus",
        provider=ModelProvider.GLM,
        model_id="glm/glm-4-plus",
        description="Zhipu's flagship model",
        context_length=128000,
    ),
    ModelInfo(
        name="GLM-4V Plus",
        provider=ModelProvider.GLM,
        model_id="glm/glm-4v-plus",
        description="Vision-enabled model",
        context_length=128000,
        supports_vision=True,
    ),
    ModelInfo(
        name="GLM-4",
        provider=ModelProvider.GLM,
        model_id="glm/glm-4",
        description="Standard GLM-4 model",
        context_length=128000,
    ),
]

KIMI_MODELS = [
    ModelInfo(
        name="Kimi k1.5",
        provider=ModelProvider.KIMI,
        model_id="moonshot/kimi-k1.5",
        description="Moonshot's latest reasoning model",
        context_length=128000,
    ),
    ModelInfo(
        name="Kimi k1.5-preview",
        provider=ModelProvider.KIMI,
        model_id="moonshot/kimi-k1.5-preview",
        description="Preview of reasoning model",
        context_length=128000,
    ),
]

LOCAL_MODELS = [
    LocalModelInfo(
        name="Llama 3 (Ollama)",
        backend="ollama",
        model_id="llama3",
        requires_gpu=True,
        description="Meta's Llama 3 via Ollama",
        memory_requirement_gb=8,
    ),
    LocalModelInfo(
        name="Llama 3.1 (Ollama)",
        backend="ollama",
        model_id="llama3.1",
        requires_gpu=True,
        description="Meta's Llama 3.1 via Ollama",
        memory_requirement_gb=8,
    ),
    LocalModelInfo(
        name="Mistral (Ollama)",
        backend="ollama",
        model_id="mistral",
        requires_gpu=True,
        description="Mistral AI model via Ollama",
        memory_requirement_gb=8,
    ),
    LocalModelInfo(
        name="Phi 3 (Ollama)",
        backend="ollama",
        model_id="phi3",
        requires_gpu=False,
        description="Microsoft's efficient model via Ollama",
        memory_requirement_gb=4,
    ),
    LocalModelInfo(
        name="Qwen 2.5 (Ollama)",
        backend="ollama",
        model_id="qwen2.5",
        requires_gpu=True,
        description="Alibaba's Qwen via Ollama",
        memory_requirement_gb=6,
    ),
    LocalModelInfo(
        name="Gemma 2 (Ollama)",
        backend="ollama",
        model_id="gemma2",
        requires_gpu=True,
        description="Google's Gemma via Ollama",
        memory_requirement_gb=8,
    ),
]

# Provider display names
PROVIDER_DISPLAY_NAMES = {
    ModelProvider.ANTHROPIC: "Anthropic (Claude)",
    ModelProvider.OPENAI: "OpenAI (GPT)",
    ModelProvider.DEEPSEEK: "DeepSeek",
    ModelProvider.MINIMAX: "MiniMax",
    ModelProvider.QWEN: "Qwen (Alibaba)",
    ModelProvider.LLAMA: "Meta Llama",
    ModelProvider.GEMINI: "Google Gemini",
    ModelProvider.GLM: "Zhipu GLM",
    ModelProvider.KIMI: "Moonshot Kimi",
    ModelProvider.LOCAL: "本地模型 (Ollama/LM Studio)",
}

# Provider API key environment variables
PROVIDER_API_KEY_VARS = {
    ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    ModelProvider.OPENAI: "OPENAI_API_KEY",
    ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    ModelProvider.MINIMAX: "MINIMAX_API_KEY",
    ModelProvider.QWEN: "DASHSCOPE_API_KEY",  # Qwen uses DashScope
    ModelProvider.LLAMA: "TOGETHER_API_KEY",  # Llama via Together AI
    ModelProvider.GEMINI: "GEMINI_API_KEY",
    ModelProvider.GLM: "ZHIPUAI_API_KEY",  # GLM uses ZhipuAI
    ModelProvider.KIMI: "MOONSHOT_API_KEY",  # Kimi uses Moonshot
}

# Provider API key environment variable (alternative names)
PROVIDER_API_KEY_ALIASES = {
    ModelProvider.ANTHROPIC: ["ANTHROPIC_KEY", "CLAUDE_API_KEY"],
    ModelProvider.OPENAI: [],
    ModelProvider.DEEPSEEK: [],
    ModelProvider.MINIMAX: [],
    ModelProvider.QWEN: ["QWEN_API_KEY"],
    ModelProvider.LLAMA: ["META_LLAMA_API_KEY"],
    ModelProvider.GEMINI: ["GOOGLE_GEMINI_API_KEY"],
    ModelProvider.GLM: [],
    ModelProvider.KIMI: [],
}

# Model catalog mapping
MODEL_CATALOG = {
    ModelProvider.ANTHROPIC: ANTHROPIC_MODELS,
    ModelProvider.OPENAI: OPENAI_MODELS,
    ModelProvider.DEEPSEEK: DEEPSEEK_MODELS,
    ModelProvider.MINIMAX: MINIMAX_MODELS,
    ModelProvider.QWEN: QWEN_MODELS,
    ModelProvider.LLAMA: LLAMA_MODELS,
    ModelProvider.GEMINI: GEMINI_MODELS,
    ModelProvider.GLM: GLM_MODELS,
    ModelProvider.KIMI: KIMI_MODELS,
    ModelProvider.LOCAL: LOCAL_MODELS,
}


class ModelWizard:
    """Interactive model configuration wizard."""

    def __init__(self):
        """Initialize the model wizard."""
        self.model_catalog = MODEL_CATALOG
        self.provider_display_names = PROVIDER_DISPLAY_NAMES

    def list_providers(self) -> list[ModelProvider]:
        """List all available providers.

        Returns:
            List of ModelProvider enums
        """
        return list(ModelProvider)

    def list_models(self, provider: ModelProvider) -> list:
        """List all models for a provider.

        Args:
            provider: The model provider

        Returns:
            List of ModelInfo or LocalModelInfo objects
        """
        return self.model_catalog.get(provider, [])

    def get_provider_display_name(self, provider: ModelProvider) -> str:
        """Get display name for a provider.

        Args:
            provider: The model provider

        Returns:
            Human-readable provider name
        """
        return self.provider_display_names.get(provider, provider.value)

    def configure(
        self,
        provider: ModelProvider,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> bool:
        """Configure a model provider.

        Args:
            provider: The model provider
            model_id: The model identifier
            api_key: Optional API key (will prompt if not provided)
            base_url: Optional base URL for custom endpoints

        Returns:
            True if configuration was successful
        """
        try:
            # Determine API key variable
            api_key_var = PROVIDER_API_KEY_VARS.get(provider, "")

            # Prepare environment variables
            env_vars = {
                "DEFAULT_MODEL": model_id,
            }

            if api_key:
                if api_key_var:
                    env_vars[api_key_var] = api_key
            elif provider != ModelProvider.LOCAL:
                # For cloud providers, API key is required
                if not api_key_var or not os.environ.get(api_key_var):
                    print(f"Warning: {api_key_var} not set")
                    return False

            if base_url:
                env_vars[f"{provider.value.upper()}_BASE_URL"] = base_url

            # Write to .env file
            write_env_file(env_vars)

            print(f"Successfully configured {self.get_provider_display_name(provider)}")
            print(f"Model: {model_id}")
            if api_key:
                print(f"API Key: {'*' * 8}{api_key[-4:]}")

            return True

        except Exception as e:
            print(f"Configuration failed: {e}")
            return False

    def get_current_config(self) -> dict:
        """Get current model configuration.

        Returns:
            Dictionary with current configuration
        """
        # Check existing environment variables
        config = {
            "provider": None,
            "model": os.environ.get("DEFAULT_MODEL", ""),
            "api_key_set": False,
        }

        # Try to detect provider from model name
        model = config["model"].lower()

        if "claude" in model:
            config["provider"] = ModelProvider.ANTHROPIC
            config["api_key_set"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
        elif "gpt" in model or "o1" in model:
            config["provider"] = ModelProvider.OPENAI
            config["api_key_set"] = bool(os.environ.get("OPENAI_API_KEY"))
        elif "deepseek" in model:
            config["provider"] = ModelProvider.DEEPSEEK
            config["api_key_set"] = bool(os.environ.get("DEEPSEEK_API_KEY"))
        elif "minimax" in model:
            config["provider"] = ModelProvider.MINIMAX
            config["api_key_set"] = bool(os.environ.get("MINIMAX_API_KEY"))
        elif "qwen" in model:
            config["provider"] = ModelProvider.QWEN
            config["api_key_set"] = bool(os.environ.get("DASHSCOPE_API_KEY"))
        elif "llama" in model or "meta" in model:
            config["provider"] = ModelProvider.LLAMA
            config["api_key_set"] = bool(os.environ.get("TOGETHER_API_KEY"))
        elif "gemini" in model:
            config["provider"] = ModelProvider.GEMINI
            config["api_key_set"] = bool(os.environ.get("GEMINI_API_KEY"))
        elif "glm" in model or "zhipu" in model:
            config["provider"] = ModelProvider.GLM
            config["api_key_set"] = bool(os.environ.get("ZHIPUAI_API_KEY"))
        elif "kimi" in model or "moonshot" in model:
            config["provider"] = ModelProvider.KIMI
            config["api_key_set"] = bool(os.environ.get("MOONSHOT_API_KEY"))
        else:
            config["provider"] = None

        return config

    def print_welcome(self):
        """Print welcome message and instructions."""
        print("=" * 60)
        print("   Paper Reader - 模型配置向导")
        print("=" * 60)
        print()
        print("支持的模型提供商:")
        print("-" * 40)
        for i, provider in enumerate(ModelProvider, 1):
            display_name = self.get_provider_display_name(provider)
            print(f"  {i}. {display_name}")
        print()

    def print_provider_models(self, provider: ModelProvider):
        """Print available models for a provider."""
        models = self.list_models(provider)
        display_name = self.get_provider_display_name(provider)

        print(f"\n{display_name} 可用模型:")
        print("-" * 40)

        for i, model in enumerate(models, 1):
            if isinstance(model, LocalModelInfo):
                print(f"  {i}. {model.name}")
                print(f"     后端: {model.backend}")
                print(f"     描述: {model.description}")
                if model.memory_requirement_gb:
                    print(f"     内存需求: ~{model.memory_requirement_gb}GB")
            else:
                print(f"  {i}. {model.name}")
                print(f"     模型ID: {model.model_id}")
                print(f"     描述: {model.description}")
                if model.context_length:
                    print(f"     上下文长度: {model.context_length:,}")
        print()


def write_env_file(env_vars: dict):
    """Write environment variables to .env file.

    Args:
        env_vars: Dictionary of environment variables to write
    """
    # Read existing .env content
    existing_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    existing_vars[key] = value

    # Merge with new values
    existing_vars.update(env_vars)

    # Write back
    with open(ENV_FILE, "w") as f:
        f.write("# Paper Reader Environment Configuration\n")
        f.write("# Generated by Model Wizard\n\n")
        for key, value in sorted(existing_vars.items()):
            # Mask API keys in output
            if "API_KEY" in key and value:
                f.write(f"{key}={value}  # (API key set)\n")
            else:
                f.write(f"{key}={value}\n")


def run_wizard():
    """Run the interactive model configuration wizard."""
    wizard = ModelWizard()
    wizard.print_welcome()

    # Select provider
    print("请选择模型提供商 (输入数字): ", end="")
    try:
        choice = int(input().strip())
        providers = wizard.list_providers()
        if choice < 1 or choice > len(providers):
            print("无效选择")
            return
        provider = providers[choice - 1]
    except (ValueError, IndexError):
        print("无效输入")
        return

    # Show available models
    wizard.print_provider_models(provider)

    # Select model
    models = wizard.list_models(provider)
    print(f"\n请选择模型 (输入数字): ", end="")
    try:
        choice = int(input().strip())
        if choice < 1 or choice > len(models):
            print("无效选择")
            return
        model = models[choice - 1]
    except (ValueError, IndexError):
        print("无效输入")
        return

    # Get API key if needed
    api_key = None
    if provider != ModelProvider.LOCAL:
        api_key_var = PROVIDER_API_KEY_VARS.get(provider, "")
        existing_key = os.environ.get(api_key_var)

        if not existing_key:
            print(f"\n请输入 {api_key_var} (直接回车跳过): ", end="")
            api_key = input().strip()
        else:
            print(f"\n检测到已存在的 API Key (已设置)")
            api_key = existing_key

    # Configure
    model_id = model.model_id if isinstance(model, ModelInfo) else model.model_id
    wizard.configure(provider, model_id, api_key if api_key else None)

    print("\n配置完成！请重启应用程序使配置生效。")


if __name__ == "__main__":
    run_wizard()
