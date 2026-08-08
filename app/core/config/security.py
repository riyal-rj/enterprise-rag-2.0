"""Authentication and rate-limiting configuration."""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings


class AuthSettings(EnvBaseSettings):
    """JWT issuance and validation configuration."""

    jwt_secret: SecretStr = Field(default=SecretStr(""))
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=60, gt=0)


class RateLimitSettings(EnvBaseSettings):
    """Per-user request/token throttling configuration."""

    rate_limit_requests: int = Field(default=20, gt=0)
    rate_limit_window_seconds: int = Field(default=60, gt=0)
    max_tokens_per_user_daily: int = Field(default=100_000, gt=0)
    auth_login_rate_limit_per_min: int = Field(default=5, gt=0)
    auth_register_rate_limit_per_hour: int = Field(default=3, gt=0)
