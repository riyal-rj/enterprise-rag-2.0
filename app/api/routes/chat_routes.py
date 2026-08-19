"""``/chat`` HTTP routes.

Thin by design: parse the request, delegate to :class:`ChatController`,
return its response. No business logic and no exception handling lives
here - see :mod:`app.rag_services.rag_service` and
:mod:`app.core.exception_handlers` respectively.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_chat_controller
from app.api.security import get_current_user
from app.controllers.chat_controller import ChatController
from app.models.auth_user import AuthenticatedUser
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatResponse:
    """Answer a question via retrieval-augmented generation. Requires login."""
    return controller.chat(user, payload)


@router.get("/history", response_model=ChatHistoryResponse)
def chat_history(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatHistoryResponse:
    """List the caller's own past questions and answers, most recent first."""
    return controller.get_history(user.username, limit, offset)


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    search: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=100, gt=0, le=200),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ConversationListResponse:
    """List the caller's own conversation threads, most recently active first."""
    return controller.list_conversations(user.username, search, limit, offset)


@router.get("/conversations/{conversation_id}/messages", response_model=ChatHistoryResponse)
def conversation_messages(
    conversation_id: int,
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatHistoryResponse:
    """Return one conversation's full thread, oldest turn first. 404 if not the caller's own."""
    return controller.get_conversation_messages(user.username, conversation_id)
