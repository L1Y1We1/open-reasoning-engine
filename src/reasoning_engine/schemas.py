from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int
    source: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    max_retries: int | None = Field(default=None, ge=0, le=3)


class QueryResponse(BaseModel):
    answer: str
    rewritten_question: str
    citations: list[Citation]
    reasoning_steps: list[str]
    retries: int


class IngestResponse(BaseModel):
    files: int
    documents: int
    collection: str


class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store: str

