"""JWT issuance, behind a swappable interface (Strategy/Adapter)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

import jwt

from app.core.exceptions import InvalidTokenError, TokenExpiredError
from app.models.auth_user import AuthenticatedUser


class TokenIssuer(Protocol):
    """Access-token issuance contract."""

    def issue(
        self,
        username: str,
        *,
        is_admin: bool = False,
        expires_delta_seconds: int | None = None,
    ) -> str: ...


class TokenVerifier(Protocol):
    """Access-token verification contract."""

    def verify(self, token: str) -> AuthenticatedUser: ...


class JWTTokenIssuer:
    """HS256 JWT :class:`TokenIssuer` backed by PyJWT."""

    def __init__(self, secret: str, algorithm: str, expires_minutes: int) -> None:
        if not secret:
            raise ValueError("JWT secret must not be empty")
        self._secret = secret
        self._algorithm = algorithm
        self._expires_minutes = expires_minutes

    def issue(
        self,
        username: str,
        *,
        is_admin: bool = False,
        expires_delta_seconds: int | None = None,
    ) -> str:
        if expires_delta_seconds is None:
            expires_delta_seconds = self._expires_minutes * 60

        now = datetime.now(UTC)
        payload = {
            "sub": username,
            "is_admin": is_admin,
            "iat": now,
            "exp": now + timedelta(seconds=expires_delta_seconds),
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)


class JWTTokenVerifier:
    """HS256 JWT :class:`TokenVerifier` backed by PyJWT."""

    def __init__(self, secret: str, algorithm: str) -> None:
        if not secret:
            raise ValueError("JWT secret must not be empty")
        self._secret = secret
        self._algorithm = algorithm

    def verify(self, token: str) -> AuthenticatedUser:
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError() from exc
        except jwt.InvalidTokenError as exc:  # PyJWT's base error, not ours
            raise InvalidTokenError() from exc

        username = payload.get("sub")
        if not username:
            raise InvalidTokenError()
        return AuthenticatedUser(username=username, is_admin=bool(payload.get("is_admin", False)))
