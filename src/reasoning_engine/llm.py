from typing import Protocol


class LanguageModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class OllamaLanguageModel:
    """Thin adapter that keeps the graph independent from a specific LLM client."""

    def __init__(self, model: str, base_url: str, timeout: float) -> None:
        from llama_index.llms.ollama import Ollama

        self._client = Ollama(model=model, base_url=base_url, request_timeout=timeout)

    def complete(self, prompt: str) -> str:
        return str(self._client.complete(prompt)).strip()

