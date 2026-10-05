"""Provider implementation for any OpenAI-compatible API (OpenAI, Groq, etc.),
using the official OpenAI SDK. Takes api_key, base_url and model in its constructor."""

from collections.abc import Iterator

from openai import OpenAI

from llm_chat_cli.models import Message
from llm_chat_cli.providers.base import Provider


class OpenAIProvider(Provider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        # base_url=None means the default OpenAI endpoint; Groq passes its own URL.
        # The SDK retries connection errors, 429 and 5xx with exponential backoff.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=3)
        self.model = model

    def stream(self, messages: list[Message], system: str) -> Iterator[str]:
        # Unlike Anthropic, the system prompt is just the first message of the list.
        payload = [{"role": "system", "content": system}]
        payload += [m.model_dump() for m in messages]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            stream=True,
        )
        for chunk in response:
            # Some chunks carry no choices or no text (e.g. the final one), so skip them.
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
