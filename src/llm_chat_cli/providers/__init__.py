"""Provider package: pluggable backends for each supported LLM API.

Will hold a factory function that, given the --model flag value
(anthropic, openai or groq), returns the correctly configured provider."""
