from functools import lru_cache

from .config import get_settings
from .engine import ReasoningEngine
from .llm import OllamaLanguageModel
from .retrieval import LlamaIndexKnowledgeBase


class AppService:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.knowledge_base = LlamaIndexKnowledgeBase(settings)
        llm = OllamaLanguageModel(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout,
        )
        self.engine = ReasoningEngine(llm, self.knowledge_base, settings)


@lru_cache
def get_service() -> AppService:
    return AppService()

