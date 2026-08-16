"""Request/response schemas for the ``/chat`` (RAG) route."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["dense", "hybrid"]


class ChatRequest(BaseModel):
    """Body for ``POST /chat``."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, gt=0, le=50)
    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        description="Overrides the server-configured default retrieval strategy for this request.",
    )
    conversation_id: int | None = Field(
        default=None,
        description="Continues an existing conversation thread; omit to start a new one.",
    )


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
    reranked: bool = False
    retrieved_chunks: list[RetrievedChunkPreview]
    flagged_claims: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response for ``POST /chat``."""

    answer: str
    sources: list[str]
    confidence: float
    cache_hit: bool = False
    metadata: ResponseMetadata
    # RAGService (retrieval/generation only) doesn't know about conversations;
    # this defaults to 0 there and is always overwritten by ChatController
    # with the real thread id before the response reaches a caller.
    conversation_id: int = 0


class ChatHistoryItem(BaseModel):
    """A single past question/answer turn, as surfaced back to its owner."""

    id: int
    question: str
    answer: str
    sources: list[str]
    confidence: float
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    """Response for ``GET /chat/history``, most recent turn first.

    Also backs ``GET /chat/conversations/{id}/messages`` - a conversation's
    full thread is the same shape, oldest turn first.
    """

    items: list[ChatHistoryItem]


class ConversationSummary(BaseModel):
    """A conversation thread, as surfaced in the sidebar's history list."""

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """Response for ``GET /chat/conversations``, most recently active first."""

    items: list[ConversationSummary]
