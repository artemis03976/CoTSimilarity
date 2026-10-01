"""Configuration management for LLM API integration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.env import load_project_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"


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
            if self.base_url is None and self.provider.lower() == "deepseek":
                self.base_url = DEFAULT_BASE_URL

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
