"""Runtime configuration for backend services."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_KIMI_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_KIMI_MODEL = "kimi-k2.6"


load_dotenv(".env.local")
load_dotenv()


@dataclass(frozen=True)
class Settings:
    voyage_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    kimi_api_key: str
    kimi_base_url: str = DEFAULT_KIMI_BASE_URL
    kimi_model: str = DEFAULT_KIMI_MODEL
    frontend_origin: str = "http://localhost:3000"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _kimi_api_key() -> str:
    value = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    if not value:
        raise RuntimeError("Missing required environment variable: KIMI_API_KEY")
    return value


def get_settings() -> Settings:
    return Settings(
        voyage_api_key=_required_env("VOYAGE_API_KEY"),
        qdrant_url=_required_env("QDRANT_URL"),
        qdrant_api_key=_required_env("QDRANT_API_KEY"),
        kimi_api_key=_kimi_api_key(),
        kimi_base_url=os.getenv("KIMI_BASE_URL", DEFAULT_KIMI_BASE_URL),
        kimi_model=os.getenv("KIMI_MODEL", DEFAULT_KIMI_MODEL),
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
    )
