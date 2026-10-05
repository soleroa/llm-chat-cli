"""System prompts and prompt templates used when talking to the LLMs."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant in a command-line chat. "
    "Answer clearly and concisely, and reply in the language the user writes in."
)


def structured_system_prompt(json_schema: str) -> str:
    return (
        "You extract structured data. Respond with ONLY a JSON object that "
        "matches this JSON Schema, with no extra text and no markdown:\n"
        f"{json_schema}"
    )
