"""Pydantic data models shared across the app (messages, conversations, config)."""

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """A single turn in a conversation.

    Only "user" and "assistant" roles live in the history: the system prompt
    is passed separately because Anthropic requires it outside the messages list.
    """

    role: Literal["user", "assistant"]
    content: str


def trim_history(history: list[Message], max_messages: int) -> None:
    """Sliding window: drop the oldest messages in place so at most `max_messages` remain.

    Never leaves an "assistant" message first, since Anthropic requires the
    conversation to start with a "user" message.
    """
    del history[: max(0, len(history) - max_messages)]
    while history and history[0].role != "user":
        history.pop(0)


class Summary(BaseModel):
    """Example structured output: what the model must return for /summary."""

    title: str
    key_points: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
