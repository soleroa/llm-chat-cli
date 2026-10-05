import pytest

from llm_chat_cli.models import Summary
from llm_chat_cli.providers.base import Provider
from llm_chat_cli.structured import ask_structured

GOOD = '{"title": "T", "key_points": ["a"], "sentiment": "positive"}'
BAD_SENTIMENT = '{"title": "T", "key_points": ["a"], "sentiment": "happy"}'


class FakeProvider(Provider):
    """Returns canned replies in order and records what it was sent."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = iter(replies)
        self.calls: list[list[str]] = []

    def stream(self, messages, system):
        self.calls.append([m.content for m in messages])
        yield next(self.replies)


def test_valid_json_parses_on_first_try():
    provider = FakeProvider([GOOD])
    result = ask_structured(provider, Summary, "summarize")
    assert result.sentiment == "positive"
    assert len(provider.calls) == 1


def test_fenced_json_is_accepted():
    provider = FakeProvider([f"```json\n{GOOD}\n```"])
    assert ask_structured(provider, Summary, "summarize").title == "T"


def test_invalid_output_is_retried_with_the_error_fed_back():
    provider = FakeProvider([BAD_SENTIMENT, GOOD])
    result = ask_structured(provider, Summary, "summarize")
    assert result.sentiment == "positive"
    assert len(provider.calls) == 2
    # The retry includes the bad answer and a message describing the problem.
    assert provider.calls[1][1] == BAD_SENTIMENT
    assert "invalid" in provider.calls[1][2]


def test_gives_up_after_max_retries():
    provider = FakeProvider(["nope"] * 3)
    with pytest.raises(ValueError):
        ask_structured(provider, Summary, "summarize", max_retries=2)
    assert len(provider.calls) == 3
