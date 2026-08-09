"""Domain-level exceptions.

Services raise these instead of framework-specific errors (e.g.
``HTTPException``), so business logic stays importable and testable without
FastAPI. The HTTP layer maps them to status codes once, centrally, via
:func:`app.core.exception_handlers.register_exception_handlers`.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all domain errors."""


class UserAlreadyExistsError(AppError):
    """Raised when registering a username that is already taken."""

    def __init__(self, username: str) -> None:
        super().__init__(f"User '{username}' already exists")
        self.username = username


class InvalidCredentialsError(AppError):
    """Raised when login credentials don't match a known user."""

    def __init__(self) -> None:
        super().__init__("Invalid username or password")


class RateLimitExceededError(AppError):
    """Raised when a caller exceeds the allotted request rate for a route."""

    def __init__(self, route: str) -> None:
        super().__init__(f"Rate limit exceeded for {route}")
        self.route = route


class InvalidTokenError(AppError):
    """Raised when a bearer token is missing, malformed, or fails verification."""

    def __init__(self, message: str = "Invalid or missing access token") -> None:
        super().__init__(message)


class TokenExpiredError(InvalidTokenError):
    """Raised when a bearer token is well-formed but past its expiry.

    Subclasses ``InvalidTokenError`` so it's covered by the same 401 handler
    (Starlette resolves exception handlers via the exception's MRO) while
    still producing a more specific message.
    """

    def __init__(self) -> None:
        super().__init__("Token has expired")


class PermissionDeniedError(AppError):
    """Raised when an authenticated caller lacks the required permission."""

    def __init__(self) -> None:
        super().__init__("You do not have permission to perform this action")
