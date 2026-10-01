"""Configuration management for LLM API integration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without dependencies installed
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _load_project_dotenv() -> None:
    """Load the project .env without overwriting already-exported variables."""
    if load_dotenv is not None:
        load_dotenv(dotenv_path=DOTENV_PATH, override=False)
        return

    # Keep config imports usable in minimal environments before requirements are
    # installed. This handles the simple KEY=value form used by the project.
    if not DOTENV_PATH.is_file():
        return
    for raw_line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


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
        _load_project_dotenv()
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

        if not self.api_key:
            raise ValueError(
                f"API key not found. Set {env_key} in the environment or .env file."
            )

    @classmethod
    def from_env(cls, provider: str = "deepseek"):
        """Create config from environment variables."""
        _load_project_dotenv()
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
        )
