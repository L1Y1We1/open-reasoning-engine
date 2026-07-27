from reasoning_engine.config import AppSettings
from reasoning_engine.engine import ReasoningEngine
from reasoning_engine.retrieval import RetrievedContext, Retriever


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedContext]:
        self.queries.append(query)
        return [
            RetrievedContext(
                text="Qdrant is the default vector database.",
                source="architecture.md",
                score=0.91,
                metadata={"page": 1},
            )
        ]


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def complete(self, prompt: str) -> str:
        return next(self.responses)


def settings() -> AppSettings:
    return AppSettings(_env_file=None, max_retries=1)


def test_pass_path_returns_citations() -> None:
    retriever = FakeRetriever()
    llm = SequenceLLM(
        ["What is the default vector database?", "The default is Qdrant. [1]", "PASS"]
    )
    response = ReasoningEngine(llm, retriever, settings()).query(
        "What is the default database?"
    )

    assert response.answer == "The default is Qdrant. [1]"
    assert response.citations[0].source == "architecture.md"
    assert response.retries == 0
    assert retriever.queries == ["What is the default vector database?"]


def test_reflection_can_retry_once() -> None:
    retriever = FakeRetriever()
    llm = SequenceLLM(
        [
            "database",
            "Uncertain",
            "RETRY: Which vector database is used by default?",
            "The default is Qdrant. [1]",
            "PASS",
        ]
    )
    response = ReasoningEngine(llm, retriever, settings()).query(
        "What is the default database?"
    )

    assert response.answer == "The default is Qdrant. [1]"
    assert response.retries == 1
    assert retriever.queries == [
        "database",
        "Which vector database is used by default?",
    ]


def test_retry_budget_zero_finalizes_without_loop() -> None:
    retriever = FakeRetriever()
    llm = SequenceLLM(
        [
            "database",
            "Uncertain",
            "RETRY: default database",
            "The evidence is insufficient to confirm the answer.",
        ]
    )
    response = ReasoningEngine(llm, retriever, settings()).query(
        "What is the default database?", max_retries=0
    )

    assert response.answer == "The evidence is insufficient to confirm the answer."
    assert response.retries == 0
    assert len(retriever.queries) == 1
