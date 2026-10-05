"""Entry point and command-line interface for the chat tool.

Supports a --model flag with three options: anthropic, openai or groq."""

import argparse

from dotenv import load_dotenv

from llm_chat_cli.models import Message
from llm_chat_cli.prompts import DEFAULT_SYSTEM_PROMPT
from llm_chat_cli.providers import SUPPORTED_PROVIDERS, create_provider

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an LLM from the terminal.")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_PROVIDERS,
        default="anthropic",
        help="which provider to talk to (default: anthropic)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    try:
        provider = create_provider(args.model)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")

    # The history lives here: the APIs are stateless, so we resend it on every turn.
    history: list[Message] = []
    print(f"Chatting with {args.model}. Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break

        history.append(Message(role="user", content=user_input))

        print("bot> ", end="", flush=True)
        reply = ""
        try:
            for chunk in provider.stream(history, DEFAULT_SYSTEM_PROMPT):
                print(chunk, end="", flush=True)
                reply += chunk
        except Exception as e:
            # Drop the unanswered user message so the history keeps alternating roles.
            history.pop()
            print(f"\n[error] {e}\n")
            continue
        print("\n")

        history.append(Message(role="assistant", content=reply))


if __name__ == "__main__":
    main()
