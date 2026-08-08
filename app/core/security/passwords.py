"""Password hashing, behind a swappable interface (Strategy/Adapter)."""

from __future__ import annotations

from typing import Protocol

from passlib.context import CryptContext


class PasswordHasher(Protocol):
    """Hashing strategy contract, so the scheme can change without touching
    callers."""

    def hash(self, password: str) -> str: ...

    def verify(self, password: str, password_hash: str) -> bool: ...


class BcryptPasswordHasher:
    """Bcrypt-backed :class:`PasswordHasher` (via passlib)."""

    def __init__(self) -> None:
        self._context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash(self, password: str) -> str:
        return self._context.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        return self._context.verify(password, password_hash)
