import anthropic
import httpx2  # HTTP library used by the current anthropic/openai SDKs
import openai
import pytest

from llm_chat_cli.providers import create_provider
from llm_chat_cli.providers.anthropic_provider import AnthropicProvider
from llm_chat_cli.providers.errors import describe_error
from llm_chat_cli.providers.openai_provider import OpenAIProvider

KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test without keys or model overrides from the real environment."""
    for var in (*KEY_VARS, "ANTHROPIC_MODEL", "OPENAI_MODEL", "GROQ_MODEL", "GROQ_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_anthropic_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(create_provider("anthropic"), AnthropicProvider)


def test_openai_provider_uses_default_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    provider = create_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    assert "api.openai.com" in str(provider.client.base_url)


def test_groq_reuses_openai_provider_with_its_own_url(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    provider = create_provider("groq")
    assert isinstance(provider, OpenAIProvider)
    assert "api.groq.com" in str(provider.client.base_url)


def test_model_can_be_overridden_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "my-model")
    assert create_provider("groq").model == "my-model"


def test_missing_key_raises_helpful_error():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_provider("openai")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("gemini")


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://example.com"))


@pytest.mark.parametrize("sdk", [anthropic, openai])
def test_describe_error_for_both_sdks(sdk):
    request = httpx2.Request("POST", "https://example.com")
    assert "timed out" in describe_error(sdk.APITimeoutError(request=request))
    assert "internet" in describe_error(sdk.APIConnectionError(request=request))
    assert "API key" in describe_error(
        sdk.AuthenticationError("x", response=_response(401), body=None)
    )
    assert "Rate limit" in describe_error(
        sdk.RateLimitError("x", response=_response(429), body=None)
    )
    assert "Model not found" in describe_error(
        sdk.NotFoundError("x", response=_response(404), body=None)
    )


def test_describe_error_falls_back_to_the_original_message():
    assert describe_error(RuntimeError("boom")) == "boom"
