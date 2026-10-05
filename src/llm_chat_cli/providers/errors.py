"""Turn SDK exceptions into short, actionable messages for the terminal."""

import anthropic
import openai

# Both SDKs share exception names; check subclasses before their parents
# (a timeout is also a connection error).
_TIMEOUT = (anthropic.APITimeoutError, openai.APITimeoutError)
_CONNECTION = (anthropic.APIConnectionError, openai.APIConnectionError)
_AUTH = (anthropic.AuthenticationError, openai.AuthenticationError)
_RATE_LIMIT = (anthropic.RateLimitError, openai.RateLimitError)
_NOT_FOUND = (anthropic.NotFoundError, openai.NotFoundError)


def describe_error(e: Exception) -> str:
    if isinstance(e, _TIMEOUT):
        return "The request timed out. Try again in a moment."
    if isinstance(e, _CONNECTION):
        return "Could not reach the API. Check your internet connection."
    if isinstance(e, _AUTH):
        return "Authentication failed. Check the API key in your .env file."
    if isinstance(e, _RATE_LIMIT):
        return "Rate limit reached (already retried automatically). Wait a bit and try again."
    if isinstance(e, _NOT_FOUND):
        return f"Model not found. Check the *_MODEL value in your .env. ({e})"
    return str(e)
