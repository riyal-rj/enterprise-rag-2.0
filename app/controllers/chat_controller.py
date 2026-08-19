"""Chat controller: schema in, schema out."""

from __future__ import annotations

import logging

from app.core.exceptions import ConversationNotFoundError
from app.models.auth_user import AuthenticatedUser
from app.models.chat_history import ChatHistoryEntry
from app.query_orchestration.query_orchestrator import QueryOrchestrator
from app.repositories.chat_history_repository import ChatHistoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.chat import (
    ChatHistoryItem,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationSummary,
)

logger = logging.getLogger(__name__)

_TITLE_MAX_LENGTH = 60


class ChatController:
    """Orchestrates :class:`QueryOrchestrator` calls for the ``/chat`` routes.

    Takes the full :class:`AuthenticatedUser` (not just ``username``, as
    before ``QueryOrchestrator`` existed) - SQL routing needs ``is_admin``
    to gate proposal creation (see ``QueryOrchestrator.answer``).
    """

    def __init__(
        self,
        query_orchestrator: QueryOrchestrator,
        chat_history_repository: ChatHistoryRepository,
        conversation_repository: ConversationRepository,
    ) -> None:
        self._query_orchestrator = query_orchestrator
        self._chat_history_repository = chat_history_repository
        self._conversation_repository = conversation_repository

    def chat(self, user: AuthenticatedUser, payload: ChatRequest) -> ChatResponse:
        username = user.username
        conversation_id = self._resolve_conversation(username, payload)
        response = self._query_orchestrator.answer(
            principal=user,
            conversation_id=conversation_id,
            question=payload.question,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
        )
        self._save_history(username, conversation_id, payload.question, response)
        return response.model_copy(update={"conversation_id": conversation_id})

    def get_history(self, username: str, limit: int, offset: int) -> ChatHistoryResponse:
        entries = self._chat_history_repository.list_by_user(username, limit=limit, offset=offset)
        return self._to_history_response(entries)

    def list_conversations(
        self, username: str, search: str | None, limit: int, offset: int
    ) -> ConversationListResponse:
        conversations = self._conversation_repository.list_by_user(username, search, limit, offset)
        return ConversationListResponse(
            items=[
                ConversationSummary(
                    id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at
                )
                for c in conversations
            ]
        )

    def get_conversation_messages(self, username: str, conversation_id: int) -> ChatHistoryResponse:
        conversation = self._conversation_repository.get_owned(conversation_id, username)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        entries = self._chat_history_repository.list_by_conversation(conversation_id)
        return self._to_history_response(entries)

    def _resolve_conversation(self, username: str, payload: ChatRequest) -> int:
        if payload.conversation_id is not None:
            conversation = self._conversation_repository.get_owned(
                payload.conversation_id, username
            )
            if conversation is None:
                raise ConversationNotFoundError(payload.conversation_id)
            return conversation.id

        conversation = self._conversation_repository.create(username, _make_title(payload.question))
        return conversation.id

    def _save_history(
        self, username: str, conversation_id: int, question: str, response: ChatResponse
    ) -> None:
        # Best-effort: the answer has already been produced (and paid for),
        # so a transient history write failure shouldn't turn a successful
        # answer into a 500 - same risk tolerance as the RAG response cache.
        try:
            self._chat_history_repository.create(
                username=username,
                conversation_id=conversation_id,
                question=question,
                answer=response.answer,
                sources=response.sources,
                confidence=response.confidence,
            )
            self._conversation_repository.touch(conversation_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "chat.history_write_failed", extra={"username": username, "error": str(exc)}
            )

    @staticmethod
    def _to_history_response(entries: list[ChatHistoryEntry]) -> ChatHistoryResponse:
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


def _make_title(question: str) -> str:
    """Derive a short conversation title from its first question."""
    stripped = " ".join(question.split())
    if len(stripped) <= _TITLE_MAX_LENGTH:
        return stripped

    truncated = stripped[:_TITLE_MAX_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return f"{truncated}…"
