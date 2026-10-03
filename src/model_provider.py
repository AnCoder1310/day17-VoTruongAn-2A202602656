from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and common aliases to canonical keys."""
    cleaned = (value or "").strip().lower().replace("-", "_")
    mapping = {
        "openai": "openai",
        "open_ai": "openai",
        "custom": "custom",
        "openai_compatible": "custom",
        "compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google_genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open_router": "openrouter",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if "openrouter" in cleaned:
        return "openrouter"
    if "ollama" in cleaned:
        return "ollama"
    if "anthropic" in cleaned or "claude" in cleaned:
        return "anthropic"
    if "gemini" in cleaned or "google" in cleaned:
        return "gemini"
    if "custom" in cleaned:
        return "custom"
    if "openai" in cleaned:
        return "openai"
    return cleaned


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": config.api_key or "custom_api_key",
            "base_url": config.base_url or "http://localhost:8000/v1",
        }
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url or "http://localhost:11434",
        }
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unsupported provider: {config.provider}")
