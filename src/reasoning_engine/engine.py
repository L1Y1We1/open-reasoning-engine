from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import AppSettings
from .llm import LanguageModel
from .prompts import ANALYZE_PROMPT, ANSWER_PROMPT, FINALIZE_PROMPT, REFLECT_PROMPT
from .retrieval import Retriever
from .schemas import Citation, QueryResponse


class ReasoningState(TypedDict, total=False):
    question: str
    rewritten_question: str
    top_k: int
    contexts: list[dict[str, Any]]
    draft: str
    critique: str
    retry_count: int
    max_retries: int
    answer: str
    reasoning_steps: list[str]
    should_retry: bool


def _format_context(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "(No relevant evidence was retrieved.)"
    blocks = []
    for index, item in enumerate(contexts, start=1):
        blocks.append(f"[{index}] Source: {item['source']}\n{item['text']}")
    return "\n\n".join(blocks)


class ReasoningEngine:
    def __init__(self, llm: LanguageModel, retriever: Retriever, settings: AppSettings) -> None:
        self.llm = llm
        self.retriever = retriever
        self.settings = settings
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ReasoningState)
        builder.add_node("analyze", self._analyze)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("answer", self._answer)
        builder.add_node("reflect", self._reflect)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "retrieve")
        builder.add_edge("retrieve", "answer")
        builder.add_edge("answer", "reflect")
        builder.add_conditional_edges(
            "reflect",
            self._route_after_reflection,
            {"retry": "retrieve", "finish": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    def _analyze(self, state: ReasoningState) -> dict[str, Any]:
        rewritten = self.llm.complete(ANALYZE_PROMPT.format(question=state["question"]))
        return {
            "rewritten_question": rewritten or state["question"],
            "reasoning_steps": ["Analyzed the question and created a retrieval plan"],
        }

    def _retrieve(self, state: ReasoningState) -> dict[str, Any]:
        contexts = self.retriever.retrieve(state["rewritten_question"], state["top_k"])
        return {
            "contexts": [item.as_dict() for item in contexts],
            "reasoning_steps": [
                *state.get("reasoning_steps", []),
                "Retrieved evidence from the vector knowledge base",
            ],
        }

    def _answer(self, state: ReasoningState) -> dict[str, Any]:
        draft = self.llm.complete(
            ANSWER_PROMPT.format(
                question=state["question"],
                rewritten_question=state["rewritten_question"],
                context=_format_context(state["contexts"]),
            )
        )
        return {
            "draft": draft,
            "reasoning_steps": [
                *state.get("reasoning_steps", []),
                "Generated a cited answer from the evidence",
            ],
        }

    def _reflect(self, state: ReasoningState) -> dict[str, Any]:
        critique = self.llm.complete(
            REFLECT_PROMPT.format(
                question=state["question"],
                rewritten_question=state["rewritten_question"],
                context=_format_context(state["contexts"]),
                draft=state["draft"],
            )
        ).strip()
        updates: dict[str, Any] = {
            "critique": critique,
            "should_retry": False,
            "reasoning_steps": [
                *state.get("reasoning_steps", []),
                "Validated the answer against the evidence",
            ],
        }
        if (
            critique.upper().startswith("RETRY:")
            and state.get("retry_count", 0) < state.get("max_retries", 0)
        ):
            refined = critique.split(":", 1)[1].strip()
            if refined:
                updates["rewritten_question"] = refined
                updates["retry_count"] = state.get("retry_count", 0) + 1
                updates["should_retry"] = True
                updates["reasoning_steps"] = [
                    *updates["reasoning_steps"],
                    "Rewrote the retrieval query from validation feedback and retried",
                ]
        return updates

    @staticmethod
    def _route_after_reflection(state: ReasoningState) -> Literal["retry", "finish"]:
        return "retry" if state.get("should_retry", False) else "finish"

    def _finalize(self, state: ReasoningState) -> dict[str, Any]:
        critique = state.get("critique", "PASS")
        if critique.upper().startswith("PASS"):
            answer = state["draft"]
        else:
            answer = self.llm.complete(
                FINALIZE_PROMPT.format(
                    question=state["question"],
                    context=_format_context(state["contexts"]),
                    draft=state["draft"],
                    critique=critique,
                )
            )
        return {
            "answer": answer,
            "reasoning_steps": [
                *state.get("reasoning_steps", []),
                "Produced the final answer",
            ],
        }

    def query(
        self, question: str, top_k: int | None = None, max_retries: int | None = None
    ) -> QueryResponse:
        result = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k or self.settings.similarity_top_k,
                "retry_count": 0,
                "max_retries": (
                    self.settings.max_retries if max_retries is None else max_retries
                ),
                "reasoning_steps": [],
            }
        )
        citations = [
            Citation(
                index=index,
                source=item["source"],
                text=item["text"][:500],
                score=item.get("score"),
                metadata=item.get("metadata", {}),
            )
            for index, item in enumerate(result.get("contexts", []), start=1)
        ]
        return QueryResponse(
            answer=result["answer"],
            rewritten_question=result["rewritten_question"],
            citations=citations,
            reasoning_steps=result.get("reasoning_steps", []),
            retries=result.get("retry_count", 0),
        )
