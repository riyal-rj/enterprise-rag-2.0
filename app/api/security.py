"""FastAPI dependencies that authenticate/authorize a request.

Replaces the old ``app.middleware.auth.require_admin``/``get_current_user``
— same job (reject unauthenticated or non-admin callers), now split into
small composable dependencies that raise domain exceptions instead of
``HTTPException`` directly, so the 401/403 mapping stays centralized in
:mod:`app.core.exception_handlers` rather than duplicated at every call site.

Uses FastAPI's ``HTTPBearer`` scheme (rather than manually parsing the
``Authorization`` header) so the OpenAPI docs get a proper "Authorize"
button. ``auto_error=False`` keeps FastAPI from raising its own 403 on a
missing header — we want a missing/invalid token to be a 401
(``InvalidTokenError``), same as a malformed one, matching the rest of this
module's status codes.
"""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.deps import get_token_verifier
from app.core.exceptions import InvalidTokenError, PermissionDeniedError
from app.core.security.tokens import TokenVerifier
from app.models.auth_user import AuthenticatedUser

_bearer_scheme = HTTPBearer(auto_error=False)


def get_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Extract the raw token from an ``Authorization: Bearer <token>`` header."""
    if credentials is None:
        raise InvalidTokenError()
    return credentials.credentials


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
