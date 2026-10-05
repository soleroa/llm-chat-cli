"""Abstract base class defining the common interface for all providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from llm_chat_cli.models import Message


class Provider(ABC):
    """Common interface every LLM backend must implement."""

    @abstractmethod
    def stream(self, messages: list[Message], system: str) -> Iterator[str]:
        """Yield the model's reply as text chunks as they arrive."""
