"""Configuration management for LLM API integration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.env import load_project_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_BASE_URLS = {
    "deepseek": DEFAULT_BASE_URL,
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
}

# Keep model selection in Python so the shell and direct CLI use the same defaults.
ANNOTATORS = {
    "deepseek": {"provider": "deepseek", "model": "deepseek-flash", "thinking": False},
    "gpt": {"provider": "openai", "model": "gpt-6-luna", "thinking": False},
    "haiku": {"provider": "anthropic", "model": "claude-haiku-5-5", "thinking": False},
    "qwen": {"provider": "dashscope", "model": "qwen3.8-flash", "thinking": False},
    "volcengine": {"provider": "volcengine", "model": "ep-20261001103240-d9m5m"},
}


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    # Provider settings
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0  # Set to 1.0 for deterministic output with temperature=0.0
    max_tokens: int = 4096
    timeout: int = 60
    thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    # Rate limiting
    requests_per_minute: int = 60

    # Batch settings
    batch_size: int = 100

    def __post_init__(self):
        """Load provider credentials from the environment if not provided."""
        load_project_dotenv(DOTENV_PATH)
        self.provider = {"claude": "anthropic", "bailian": "dashscope"}.get(
            self.provider.lower(), self.provider.lower()
        )
        provider_prefix = self.provider.upper()
        env_key = f"{provider_prefix}_API_KEY"
        env_url = f"{provider_prefix}_BASE_URL"

        if self.api_key is None:
            self.api_key = os.getenv(env_key) or os.getenv("LLM_API_KEY")
        if self.base_url is None:
            self.base_url = (
                os.getenv(env_url)
                or os.getenv(f"{provider_prefix}_API_URL")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("LLM_API_URL")
            )
            if self.base_url is None:
                self.base_url = DEFAULT_BASE_URLS.get(self.provider)

    @classmethod
    def for_annotator(cls, annotator: str, **overrides):
        settings = dict(ANNOTATORS[annotator])
        settings.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**settings)

    @classmethod
    def from_env(cls, provider: str = "deepseek"):
        """Create config from environment variables."""
        load_project_dotenv(DOTENV_PATH)
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
        )
