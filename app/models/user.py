"""Internal ``User`` domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    """A persisted user record."""

    id: int
    username: str
    password_hash: str
    is_admin: bool
