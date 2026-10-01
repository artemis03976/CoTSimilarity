"""Environment-file loading used by command-line configuration."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without dependencies installed
    load_dotenv = None


def load_project_dotenv(dotenv_path) -> None:
    """Load a dotenv file without replacing exported environment values."""
    dotenv_path = Path(dotenv_path)
    if load_dotenv is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    # Keep config imports usable in minimal environments before requirements are
    # installed. This handles the simple KEY=value form used by the project.
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
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


