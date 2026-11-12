"""Load local provider settings without replacing exported environment values."""

from dotenv import load_dotenv


def load_project_dotenv(dotenv_path) -> None:
    load_dotenv(dotenv_path=dotenv_path, override=False)
