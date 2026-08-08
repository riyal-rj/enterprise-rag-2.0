"""JWT issuance, behind a swappable interface (Strategy/Adapter)."""

from __future__ import annotations

import time
from typing import Protocol

import jwt

from app.core.exceptions import InvalidTokenError
from app.models.auth_user import AuthenticatedUser


class TokenIssuer(Protocol):
    """Access-token issuance contract."""

    def issue(self, username: str, *, is_admin: bool = False) -> str: ...


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

    def issue(self, username: str, *, is_admin: bool = False) -> str:
        now = int(time.time())
        payload = {
            "sub": username,
            "is_admin": is_admin,
            "iat": now,
            "exp": now + self._expires_minutes * 60,
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
        except jwt.PyJWTError as exc:
            raise InvalidTokenError() from exc

        username = payload.get("sub")
        if not username:
            raise InvalidTokenError()
        return AuthenticatedUser(username=username, is_admin=bool(payload.get("is_admin", False)))
