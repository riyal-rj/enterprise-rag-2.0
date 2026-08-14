"""``/chat`` HTTP routes.

Thin by design: parse the request, delegate to :class:`ChatController`,
return its response. No business logic and no exception handling lives
here - see :mod:`app.rag_services.rag_service` and
:mod:`app.core.exception_handlers` respectively.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_controller
from app.api.security import get_current_user
from app.controllers.chat_controller import ChatController
from app.models.auth_user import AuthenticatedUser
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    _: AuthenticatedUser = Depends(get_current_user),
    controller: ChatController = Depends(get_chat_controller),
) -> ChatResponse:
    """Answer a question via retrieval-augmented generation. Requires login."""
    return controller.chat(payload)
