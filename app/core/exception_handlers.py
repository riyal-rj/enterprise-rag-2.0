"""Central mapping from domain exceptions to HTTP responses.

Registered once in :func:`app.main.create_app`, so controllers can raise
plain domain exceptions (see :mod:`app.core.exceptions`) without importing
FastAPI or repeating try/except blocks per endpoint.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    ConversationNotFoundError,
    FileTooLargeError,
    HybridRetrievalDisabledError,
    InvalidCredentialsError,
    InvalidRagOpsConfigError,
    InvalidTokenError,
    PermissionDeniedError,
    RateLimitExceededError,
    UnsupportedFileTypeError,
    UserAlreadyExistsError,
)

_STATUS_BY_EXCEPTION: dict[type[AppError], int] = {
    UserAlreadyExistsError: status.HTTP_409_CONFLICT,
    InvalidCredentialsError: status.HTTP_401_UNAUTHORIZED,
    InvalidTokenError: status.HTTP_401_UNAUTHORIZED,
    PermissionDeniedError: status.HTTP_403_FORBIDDEN,
    RateLimitExceededError: status.HTTP_429_TOO_MANY_REQUESTS,
    UnsupportedFileTypeError: status.HTTP_400_BAD_REQUEST,
    FileTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    ConversationNotFoundError: status.HTTP_404_NOT_FOUND,
    HybridRetrievalDisabledError: status.HTTP_400_BAD_REQUEST,
    InvalidRagOpsConfigError: status.HTTP_400_BAD_REQUEST,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach one HTTP-mapping handler per known domain exception type."""
    for exc_type, status_code in _STATUS_BY_EXCEPTION.items():
        app.add_exception_handler(exc_type, _make_handler(status_code))


def _make_handler(status_code: int,) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        # exc may be a subclass of Exception (AppError); coerce to str for the response
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return _handler
