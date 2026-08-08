"""Structured logging configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.core.config.base import EnvBaseSettings

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingSettings(EnvBaseSettings):
    """Controls log output format and verbosity."""

    log_json: bool = Field(default=False)
    log_level: LogLevel = Field(default="INFO")
