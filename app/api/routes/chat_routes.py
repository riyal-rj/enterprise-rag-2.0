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
from app.schemas.chat import ChatHistoryResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatResponse:
    """Answer a question via retrieval-augmented generation. Requires login."""
    return controller.chat(user.username, payload)


@router.get("/history", response_model=ChatHistoryResponse)
def chat_history(
    limit: int = Query(default=20, gt=0, le=100),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatHistoryResponse:
    """List the caller's own past questions and answers, most recent first."""
    return controller.get_history(user.username, limit, offset)
