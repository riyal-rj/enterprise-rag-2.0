"""Request/response schemas for the ``/auth`` routes."""

from __future__ import annotations

from pydantic import BaseModel, Field

_USERNAME_PATTERN = r"^[A-Za-z0-9_.-]+$"


class RegisterRequest(BaseModel):
    """Body for ``POST /auth/register``."""

    username: str = Field(min_length=3, max_length=32, pattern=_USERNAME_PATTERN)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Body for ``POST /auth/login``."""

    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """Response for both auth routes."""

    token: str
    token_type: str = "bearer"
