"""Chat controller: schema in, schema out."""

from __future__ import annotations

from app.rag_services.rag_service import RAGService
from app.schemas.chat import ChatRequest, ChatResponse


class ChatController:
    """Orchestrates :class:`RAGService` calls for the ``/chat`` routes."""

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service

    def chat(self, payload: ChatRequest) -> ChatResponse:
        return self._rag_service.answer(payload.question, top_k=payload.top_k)
