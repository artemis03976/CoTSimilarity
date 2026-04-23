"""Configuration management for LLM API integration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    # Provider settings
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_key: Optional[str] = "sk-dcda9ad70d554b409d6f13ad56b14a3f"
    base_url: Optional[str] = "https://api.deepseek.com"

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
        """Load API key from environment if not provided."""
        if self.api_key is None:
            env_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)
            if not self.api_key:
                raise ValueError(
                    f"API key not found. Set {env_key} environment variable."
                )

    @classmethod
    def from_env(cls, provider: str = "deepseek"):
        """Create config from environment variables."""
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
        )
