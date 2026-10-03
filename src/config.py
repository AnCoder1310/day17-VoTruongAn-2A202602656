from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Attributes:
        base_dir: Root directory of the repository.
        data_dir: Directory containing benchmark datasets.
        state_dir: Directory storing persistent agent state (e.g. User.md).
        compact_threshold_tokens: Token threshold to trigger compact memory.
        compact_keep_messages: Number of recent messages to keep uncompressed.
        model: Main model configuration.
        judge_model: Judge model configuration for evaluation.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _get_provider_credentials(provider: str) -> tuple[str | None, str | None]:
    """Retrieve API key and base URL from environment variables for a given provider."""
    normalized = normalize_provider(provider)
    api_key: str | None = None
    base_url: str | None = None

    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif normalized == "custom":
        api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
    elif normalized == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    elif normalized == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
    elif normalized == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif normalized == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL")

    return api_key, base_url


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig instance."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    main_provider = os.getenv("LLM_PROVIDER", "openai")
    main_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    main_temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    main_key, main_url = _get_provider_credentials(main_provider)

    model_config = ProviderConfig(
        provider=main_provider,
        model_name=main_model,
        temperature=main_temp,
        api_key=main_key,
        base_url=main_url,
    )

    judge_provider = os.getenv("JUDGE_PROVIDER", main_provider)
    judge_model = os.getenv("JUDGE_MODEL", main_model)
    judge_temp = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    judge_key, judge_url = _get_provider_credentials(judge_provider)

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model,
        temperature=judge_temp,
        api_key=judge_key,
        base_url=judge_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config,
    )
