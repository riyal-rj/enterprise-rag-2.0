"""Chat controller: schema in, schema out."""

from __future__ import annotations

import logging

from app.rag_services.rag_service import RAGService
from app.repositories.chat_history_repository import ChatHistoryRepository
from app.schemas.chat import ChatHistoryItem, ChatHistoryResponse, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatController:
    """Orchestrates :class:`RAGService` calls for the ``/chat`` routes."""

    def __init__(
        self,
        rag_service: RAGService,
        chat_history_repository: ChatHistoryRepository,
    ) -> None:
        self._rag_service = rag_service
        self._chat_history_repository = chat_history_repository

    def chat(self, username: str, payload: ChatRequest) -> ChatResponse:
        response = self._rag_service.answer(payload.question, top_k=payload.top_k)
        self._save_history(username, payload.question, response)
        return response

    def get_history(self, username: str, limit: int, offset: int) -> ChatHistoryResponse:
        entries = self._chat_history_repository.list_by_user(username, limit=limit, offset=offset)
        return ChatHistoryResponse(
            items=[
                ChatHistoryItem(
                    id=entry.id,
                    question=entry.question,
                    answer=entry.answer,
                    sources=entry.sources,
                    confidence=entry.confidence,
                    created_at=entry.created_at,
                )
                for entry in entries
            ]
        )

    def _save_history(self, username: str, question: str, response: ChatResponse) -> None:
        # Best-effort: the answer has already been produced (and paid for),
        # so a transient history write failure shouldn't turn a successful
        # answer into a 500 - same risk tolerance as the RAG response cache.
        try:
            self._chat_history_repository.create(
                username=username,
                question=question,
                answer=response.answer,
                sources=response.sources,
                confidence=response.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat.history_write_failed", extra={"username": username, "error": str(exc)})
