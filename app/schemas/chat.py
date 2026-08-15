"""Request/response schemas for the ``/chat`` (RAG) route."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body for ``POST /chat``."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, gt=0, le=50)


class RetrievedChunkPreview(BaseModel):
    """A chunk as surfaced back to the caller, alongside the answer."""

    text: str
    source: str
    score: float
    page_number: int | None = None


class ResponseMetadata(BaseModel):
    """Diagnostic detail about how an answer was produced.

    ``route`` is fixed to ``"rag"`` for now; it exists so later routing
    (SQL / hybrid intents) can be added without changing the response
    shape callers already depend on.
    """

    route: str
    retrieval_mode:str = "dense"
    retrieved_chunks: list[RetrievedChunkPreview]
    flagged_claims: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response for ``POST /chat``."""

    answer: str
    sources: list[str]
    confidence: float
    cache_hit: bool = False
    metadata: ResponseMetadata


class ChatHistoryItem(BaseModel):
    """A single past question/answer turn, as surfaced back to its owner."""

    id: int
    question: str
    answer: str
    sources: list[str]
    confidence: float
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Response for ``GET /chat/history``, most recent turn first."""

    items: list[ChatHistoryItem]
