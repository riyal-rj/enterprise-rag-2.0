"""Password hashing, behind a swappable interface (Strategy/Adapter)."""

from __future__ import annotations

from typing import Protocol

import bcrypt

_DEFAULT_ROUNDS = 12


class PasswordHasher(Protocol):
    """Hashing strategy contract, so the scheme can change without touching
    callers."""

    def hash(self, password: str) -> str: ...

    def verify(self, password: str, password_hash: str) -> bool: ...


class BcryptPasswordHasher:
    """Bcrypt :class:`PasswordHasher` backed directly by the ``bcrypt`` library."""

    def __init__(self, rounds: int = _DEFAULT_ROUNDS) -> None:
        self._rounds = rounds

    def hash(self, password: str) -> str:
        salt = bcrypt.gensalt(rounds=self._rounds)
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    def verify(self, password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
