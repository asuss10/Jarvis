"""Application settings, loaded once from environment variables / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# config.py -> jarvis/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    model: str
    num_ctx: int
    log_level: str
    data_dir: Path


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("JARVIS_MODEL", "gpt-oss:20b"),
        num_ctx=int(os.getenv("JARVIS_NUM_CTX", "8192")),
        log_level=os.getenv("JARVIS_LOG_LEVEL", "INFO").upper(),
        data_dir=PROJECT_ROOT / "data",
    )