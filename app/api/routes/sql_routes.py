"""``/sql-proposals`` HTTP routes: approve or reject a pending SQL proposal.

Admin-only for this rollout (see ``require_admin``), matching the same gate
``QueryOrchestrator`` applies before ever creating a proposal. The only
input is the opaque proposal id plus the authenticated principal - no SQL
text is ever accepted here; the server re-validates and executes the
immutable stored proposal (see ``app.sql.sql_service.SQLService
.approve_and_execute``).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response

from app.api.deps import get_sql_controller
from app.api.security import require_admin
from app.controllers.sql_controller import SQLController
from app.models.auth_user import AuthenticatedUser
from app.schemas.chat import ChatResponse

router = APIRouter(prefix="/sql-proposals", tags=["sql"])


@router.post("/{proposal_id}/approve", response_model=ChatResponse)
def approve_sql_proposal(
    proposal_id: UUID,
    user: AuthenticatedUser = Depends(require_admin),
    controller: SQLController = Depends(get_sql_controller),
) -> ChatResponse:
    """Approve and execute a pending SQL proposal. Admin-only."""
    return controller.approve(user, proposal_id)


@router.post("/{proposal_id}/reject", status_code=status.HTTP_204_NO_CONTENT)
def reject_sql_proposal(
    proposal_id: UUID,
    user: AuthenticatedUser = Depends(require_admin),
    controller: SQLController = Depends(get_sql_controller),
) -> Response:
    """Reject a pending SQL proposal. Admin-only."""
    controller.reject(user, proposal_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
