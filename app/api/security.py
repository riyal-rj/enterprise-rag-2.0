"""FastAPI dependencies that authenticate/authorize a request.

Replaces the old ``app.middleware.auth.require_admin`` — same job (reject
unauthenticated or non-admin callers), now split into small composable
dependencies instead of one middleware function.
"""

from __future__ import annotations

from fastapi import Depends, Header

from app.api.deps import get_token_verifier
from app.core.exceptions import InvalidTokenError, PermissionDeniedError
from app.core.security.tokens import TokenVerifier
from app.models.auth_user import AuthenticatedUser

_BEARER_PREFIX = "bearer "


def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    """Extract the raw token from an ``Authorization: Bearer <token>`` header."""
    if not authorization or not authorization.lower().startswith(_BEARER_PREFIX):
        raise InvalidTokenError()
    return authorization[len(_BEARER_PREFIX) :]


def get_current_user(
    token: str = Depends(get_bearer_token),
    verifier: TokenVerifier = Depends(get_token_verifier),
) -> AuthenticatedUser:
    return verifier.verify(token)


def require_admin(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if not user.is_admin:
        raise PermissionDeniedError()
    return user
