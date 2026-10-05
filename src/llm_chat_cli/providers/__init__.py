"""Provider package: pluggable backends for each supported LLM API.

Holds a factory function that, given the --model flag value
(anthropic, openai or groq), returns the correctly configured provider."""

import os

from llm_chat_cli.providers.anthropic_provider import AnthropicProvider
from llm_chat_cli.providers.base import Provider
from llm_chat_cli.providers.openai_provider import OpenAIProvider

SUPPORTED_PROVIDERS = ("anthropic", "openai", "groq")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing {name}. Copy .env.example to .env and set it.")
    return value


def create_provider(name: str) -> Provider:
    """Build the provider for `name`, reading credentials from the environment."""
    if name == "anthropic":
        return AnthropicProvider(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
        )
    if name == "openai":
        return OpenAIProvider(
            api_key=_require_env("OPENAI_API_KEY"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    if name == "groq":
        # Groq speaks the OpenAI protocol, so we reuse OpenAIProvider with its own URL.
        return OpenAIProvider(
            api_key=_require_env("GROQ_API_KEY"),
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(SUPPORTED_PROVIDERS)}")
