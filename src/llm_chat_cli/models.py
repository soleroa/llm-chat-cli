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
