"""Provider implementation that talks to Claude via the official Anthropic SDK."""

from collections.abc import Iterator

from anthropic import Anthropic

from llm_chat_cli.models import Message
from llm_chat_cli.providers.base import Provider


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024) -> None:
        # The SDK retries connection errors, 429 and 5xx with exponential backoff.
        self.client = Anthropic(api_key=api_key, timeout=30.0, max_retries=3)
        self.model = model
        # Anthropic requires max_tokens on every request (OpenAI makes it optional).
        self.max_tokens = max_tokens

    def stream(self, messages: list[Message], system: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,  # separate parameter, not part of `messages`
            messages=[m.model_dump() for m in messages],
        ) as stream:
            yield from stream.text_stream
