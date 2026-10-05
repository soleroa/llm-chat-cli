# llm-chat-cli

A command-line tool for chatting with LLMs (Anthropic, OpenAI and Groq) using only the official SDKs, with no frameworks such as LangChain.

## Description

A terminal chat that:

- keeps a **multi-turn conversation** (the history is managed by hand, with a sliding window so it never outgrows the context),
- shows the reply with **streaming**, token by token,
- switches provider with a flag: `--model anthropic|openai|groq`,
- returns **structured output validated with Pydantic** (the `/summary` command), retrying when the model answers with invalid JSON,
- shows clear error messages (invalid key, rate limit, no connection) and retries with exponential backoff.

**Stack:** Python 3.11+, the official `anthropic` and `openai` SDKs, `pydantic`, `python-dotenv`, `pytest`.

## Installation

```bash
git clone https://github.com/soleroa/llm-chat-cli.git
cd llm-chat-cli

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # then fill in at least one API key
```

Variables in `.env`:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` | credentials (you only need the one for the provider you use) |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GROQ_MODEL` | model to use with each provider |
| `GROQ_BASE_URL` | Groq endpoint (OpenAI-compatible) |

> Model names change often. If you see "Model not found", check the `*_MODEL` value in your `.env`.

## Usage

```bash
llm-chat-cli --model groq
llm-chat-cli --model openai --max-history 10
```

| Flag | Description |
|---|---|
| `--model` | `anthropic` (default), `openai` or `groq` |
| `--max-history` | maximum number of messages remembered; older ones are dropped (default: 20) |

Commands inside the chat:

| Command | What it does |
|---|---|
| `/summary <text>` | summarizes the text as JSON validated with Pydantic (title, key points, sentiment) |
| `exit`, `quit`, Ctrl+C | quit |

### Demo

A real session with Groq:

```text
you> Me llamo Sole y estoy aprendiendo a programar con LLMs.
bot> ¡Hola Sole! 👋 Qué bueno que estés empezando a programar con LLMs. ¿En qué lenguaje
     o framework te estás enfocando?

you> ¿Cómo me llamo y qué estoy aprendiendo? Respondé en una frase.
bot> Te llamas Sole y estás aprendiendo a programar con LLMs.

you> /summary El nuevo restaurante del barrio tiene platos ricos y precios justos, aunque el servicio fue lento y tuvimos que esperar mucho.
{
  "title": "Resumen",
  "key_points": [
    "El nuevo restaurante del barrio ofrece platos ricos y precios justos",
    "El servicio fue lento y se tuvo que esperar mucho"
  ],
  "sentiment": "neutral"
}
```

The second answer shows the conversation memory at work, and `/summary` shows the validated structured output. The assistant replies in the language you write in.

### Tests

```bash
pytest
```

The tests need no network access or API keys: they use a fake provider and hand-built exceptions.

## Architecture

```
src/llm_chat_cli/
├── cli.py                  # arguments, chat loop, history
├── models.py               # Message, Summary, trim_history
├── prompts.py              # system prompts
├── structured.py           # ask_structured: JSON + validation + retry
└── providers/
    ├── base.py             # Provider interface: stream(messages, system)
    ├── anthropic_provider.py
    ├── openai_provider.py  # serves both OpenAI and Groq
    ├── errors.py           # SDK exceptions -> clear messages
    └── __init__.py         # create_provider(name): the factory
```

The core idea: `cli.py` does not know which provider it is talking to. It only uses the `Provider.stream()` interface, which yields the text in chunks as it arrives, and each provider translates that to its own SDK. Adding a new one only takes implementing that interface and registering it in the factory.

Groq reuses the OpenAI adapter (`OpenAIProvider`), since its API is OpenAI-compatible: only the key, the model and the `base_url` change.

Memory is a list of `Message` objects that is resent in full on every turn, because the APIs keep no state. `trim_history` cuts it as a sliding window and never leaves an `assistant` message first, since Anthropic requires the conversation to start with `user`.

## Differences between providers

| | Anthropic | OpenAI | Groq |
|---|---|---|---|
| System prompt | separate `system` parameter | message with `role: "system"` inside `messages` | same as OpenAI |
| `max_tokens` | required | optional | optional |
| Streaming | `client.messages.stream()` + `text_stream` | `create(stream=True)` and iterate over chunks | same as OpenAI |
| Response text | `response.content[0].text` | `response.choices[0].message.content` | same as OpenAI |
| Text of each chunk | already a `str` | `chunk.choices[0].delta.content` (can be `None`) | same as OpenAI |
| First message | must be `user` | no such restriction | same as OpenAI |
| Client in this project | `AnthropicProvider` | `OpenAIProvider` | `OpenAIProvider` with a different `base_url` |

## Status

Tested against the real **Groq** API. `AnthropicProvider` and the OpenAI setup have not been tested against their real APIs yet (only through the test suite and a fake provider).

## Possible improvements

- Trim the history by tokens, or summarize the old part, instead of counting messages.
- A circuit breaker, to stop retrying when the service is down.
- Save and resume conversations.
