import pytest
from pydantic import ValidationError

from llm_chat_cli.models import Message, Summary, trim_history


def make_history(n: int) -> list[Message]:
    """Alternating user/assistant messages, starting with user."""
    return [
        Message(role="user" if i % 2 == 0 else "assistant", content=str(i))
        for i in range(n)
    ]


def contents(history: list[Message]) -> list[str]:
    return [m.content for m in history]


def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="system", content="hi")


def test_summary_rejects_unknown_sentiment():
    with pytest.raises(ValidationError):
        Summary(title="t", key_points=[], sentiment="happy")


def test_trim_history_under_limit_changes_nothing():
    history = make_history(5)
    trim_history(history, 20)
    assert contents(history) == ["0", "1", "2", "3", "4"]


def test_trim_history_keeps_last_n_when_cut_lands_on_user():
    history = make_history(7)
    trim_history(history, 5)
    assert contents(history) == ["2", "3", "4", "5", "6"]


def test_trim_history_never_starts_with_assistant():
    history = make_history(7)
    trim_history(history, 4)  # naive cut would start at message 3 (assistant)
    assert contents(history) == ["4", "5", "6"]
    assert history[0].role == "user"


def test_trim_history_zero_empties_history():
    history = make_history(3)
    trim_history(history, 0)
    assert history == []
