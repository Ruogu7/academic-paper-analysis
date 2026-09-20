"""Tests for model configuration wizard."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.model_wizard import (
    ModelWizard,
    ModelInfo,
    ModelProvider,
    LocalModelInfo,
)


class TestModelProvider:
    """Test ModelProvider enum."""

    def test_all_providers_defined(self):
        """Test that all major providers are defined."""
        providers = [p.value for p in ModelProvider]
        expected = ["anthropic", "openai", "deepseek", "minimax", "qwen",
                    "llama", "gemini", "glm", "kimi", "local"]

        for exp in expected:
            assert exp in providers


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo object."""
        model = ModelInfo(
            name="GPT-4",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4",
            requires_api_key=True,
            description="OpenAI's GPT-4 model",
        )

        assert model.name == "GPT-4"
        assert model.provider == ModelProvider.OPENAI
        assert model.model_id == "gpt-4"
        assert model.requires_api_key is True


class TestModelWizardInterface:
    """Test ModelWizard interface."""

    def test_wizard_has_list_providers_method(self):
        """Test that wizard has list_providers method."""
        wizard = ModelWizard()
        assert hasattr(wizard, "list_providers")
        assert callable(getattr(wizard, "list_providers"))

    def test_wizard_has_list_models_method(self):
        """Test that wizard has list_models method."""
        wizard = ModelWizard()
        assert hasattr(wizard, "list_models")
        assert callable(getattr(wizard, "list_models"))

    def test_wizard_has_configure_method(self):
        """Test that wizard has configure method."""
        wizard = ModelWizard()
        assert hasattr(wizard, "configure")
        assert callable(getattr(wizard, "configure"))

    def test_wizard_has_get_current_config_method(self):
        """Test that wizard has get_current_config method."""
        wizard = ModelWizard()
        assert hasattr(wizard, "get_current_config")
        assert callable(getattr(wizard, "get_current_config"))


class TestModelWizardSupportedModels:
    """Test ModelWizard supported models list."""

    def test_all_major_providers_have_models(self):
        """Test that all major providers have models listed."""
        wizard = ModelWizard()
        providers = wizard.list_providers()

        for provider in providers:
            models = wizard.list_models(provider)
            assert len(models) > 0, f"Provider {provider} should have models"

    def test_anthropic_models_exist(self):
        """Test that Anthropic models are listed."""
        wizard = ModelWizard()
        models = wizard.list_models(ModelProvider.ANTHROPIC)

        model_ids = [m.model_id for m in models]
        assert any("claude" in m.lower() for m in model_ids)

    def test_openai_models_exist(self):
        """Test that OpenAI models are listed."""
        wizard = ModelWizard()
        models = wizard.list_models(ModelProvider.OPENAI)

        model_ids = [m.model_id for m in models]
        assert any("gpt" in m.lower() for m in model_ids)

    def test_deepseek_models_exist(self):
        """Test that DeepSeek models are listed."""
        wizard = ModelWizard()
        models = wizard.list_models(ModelProvider.DEEPSEEK)

        assert len(models) > 0

    def test_local_models_exist(self):
        """Test that local/Ollama models are listed."""
        wizard = ModelWizard()
        models = wizard.list_models(ModelProvider.LOCAL)

        assert len(models) > 0
        # Should include Ollama
        local_info = models[0]
        assert isinstance(local_info, LocalModelInfo)


class TestModelWizardConfiguration:
    """Test ModelWizard configuration."""

    def test_configure_saves_to_env(self):
        """Test that configure saves configuration to environment."""
        wizard = ModelWizard()

        with patch("src.config.model_wizard.write_env_file") as mock_write:
            result = wizard.configure(
                provider=ModelProvider.OPENAI,
                model_id="gpt-4o",
                api_key="test-key-123",
            )

            assert result is True
            mock_write.assert_called_once()

    def test_configure_with_invalid_provider(self):
        """Test configure with invalid provider."""
        wizard = ModelWizard()

        result = wizard.configure(
            provider="invalid",
            model_id="test",
            api_key="test-key",
        )

        assert result is False

    def test_get_current_config_returns_dict(self):
        """Test get_current_config returns configuration."""
        wizard = ModelWizard()
        config = wizard.get_current_config()

        assert isinstance(config, dict)
        assert "provider" in config
        assert "model" in config


class TestLocalModelInfo:
    """Test LocalModelInfo dataclass."""

    def test_local_model_info_creation(self):
        """Test creating a LocalModelInfo object."""
        model = LocalModelInfo(
            name="Llama 3",
            backend="ollama",
            model_id="llama3",
            requires_gpu=True,
            description="Meta's Llama 3 model via Ollama",
        )

        assert model.name == "Llama 3"
        assert model.backend == "ollama"
        assert model.model_id == "llama3"
        assert model.requires_gpu is True
