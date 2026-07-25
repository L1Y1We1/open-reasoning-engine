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
    llm = SequenceLLM(["默认向量数据库是什么", "默认是 Qdrant。[1]", "PASS"])
    response = ReasoningEngine(llm, retriever, settings()).query("默认数据库？")

    assert response.answer == "默认是 Qdrant。[1]"
    assert response.citations[0].source == "architecture.md"
    assert response.retries == 0
    assert retriever.queries == ["默认向量数据库是什么"]


def test_reflection_can_retry_once() -> None:
    retriever = FakeRetriever()
    llm = SequenceLLM(
        [
            "数据库",
            "不确定",
            "RETRY: 默认使用哪个向量数据库",
            "默认是 Qdrant。[1]",
            "PASS",
        ]
    )
    response = ReasoningEngine(llm, retriever, settings()).query("默认数据库？")

    assert response.answer == "默认是 Qdrant。[1]"
    assert response.retries == 1
    assert retriever.queries == ["数据库", "默认使用哪个向量数据库"]


def test_retry_budget_zero_finalizes_without_loop() -> None:
    retriever = FakeRetriever()
    llm = SequenceLLM(
        ["数据库", "不确定", "RETRY: 默认数据库", "证据不足，无法确认。"]
    )
    response = ReasoningEngine(llm, retriever, settings()).query(
        "默认数据库？", max_retries=0
    )

    assert response.answer == "证据不足，无法确认。"
    assert response.retries == 0
    assert len(retriever.queries) == 1

