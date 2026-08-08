"""Deployment environment enum and its environment-variable source."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from app.core.config.base import EnvBaseSettings


class Environment(str, Enum):
    """Deployment environment; gates strict production-only validation."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(EnvBaseSettings):
    """Loads ``ENVIRONMENT`` in isolation from the other settings groups."""

    environment: Environment = Field(default=Environment.DEVELOPMENT)
