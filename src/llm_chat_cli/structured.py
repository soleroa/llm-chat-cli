"""Structured output: ask a provider for JSON and validate it with Pydantic."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from llm_chat_cli.models import Message
from llm_chat_cli.prompts import structured_system_prompt
from llm_chat_cli.providers.base import Provider


def _extract_json(text: str) -> str:
    """Strip markdown code fences that models often wrap around JSON."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


T = TypeVar("T", bound=BaseModel)


def ask_structured(
    provider: Provider, schema: type[T], instruction: str, max_retries: int = 2
) -> T:
    """Return `instruction`'s answer as a validated `schema` instance.

    On invalid output, the error is sent back to the model so it can fix it.
    Raises ValueError if it still fails after `max_retries` retries.
    """
    system = structured_system_prompt(json.dumps(schema.model_json_schema()))
    messages = [Message(role="user", content=instruction)]

    for attempt in range(max_retries + 1):
        raw = "".join(provider.stream(messages, system))
        try:
            return schema.model_validate_json(_extract_json(raw))
        except ValidationError as e:
            if attempt == max_retries:
                raise ValueError(f"Model output never matched the schema: {e}") from e
            messages += [
                Message(role="assistant", content=raw),
                Message(
                    role="user",
                    content=f"That was invalid: {e}\nReturn only the corrected JSON.",
                ),
            ]
    raise AssertionError("unreachable")
