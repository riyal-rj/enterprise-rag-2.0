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


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded policy document isn't a type the ingestion pipeline handles."""

    def __init__(self, filename: str, supported_suffixes: set[str]) -> None:
        suffixes = ", ".join(sorted(supported_suffixes))
        super().__init__(f"'{filename}' is not a supported file type (expected one of: {suffixes})")
        self.filename = filename


class FileTooLargeError(AppError):
    """Raised when an uploaded policy document exceeds the configured size limit."""

    def __init__(self, filename: str, max_size_mb: int) -> None:
        super().__init__(f"'{filename}' exceeds the maximum upload size of {max_size_mb}MB")
        self.filename = filename


class ConversationNotFoundError(AppError):
    """Raised when a conversation doesn't exist or isn't owned by the caller.

    Deliberately doesn't distinguish the two cases in the message - the 404
    shouldn't leak whether a given id belongs to someone else.
    """

    def __init__(self, conversation_id: int) -> None:
        super().__init__(f"Conversation {conversation_id} not found")
        self.conversation_id = conversation_id


class InvalidRagOpsConfigError(AppError):
    """Raised when a RAG Operations panel config change would put the
    pipeline in an invalid state (e.g. switching to the Voyage reranker
    backend with no API key configured for this deployment)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class EvaluationPipelineError(AppError):
    """Raised when the offline eval harness detects that the system under
    test didn't actually do what a case's ``PipelineProfile`` asked for
    (e.g. HyDE was requested but bypassed/fell back) - lets a case fail
    loudly instead of silently scoring degraded/baseline output as if it
    were the feature under test. See ``app.eval.invokers.ServiceInvoker``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SQLProposalNotFoundError(AppError):
    """Raised when a SQL proposal doesn't exist or isn't owned by the caller.

    Deliberately doesn't distinguish the two cases in the message, same
    reasoning as ``ConversationNotFoundError``.
    """

    def __init__(self, proposal_id: object) -> None:
        super().__init__(f"SQL proposal {proposal_id} not found")
        self.proposal_id = proposal_id


class SQLProposalStateError(AppError):
    """Raised when a SQL proposal can't transition the way a caller asked -
    already consumed by a concurrent approval, expired, or its catalog/
    policy/fingerprint no longer matches what was approved. See
    ``app.repositories.sql_proposal_repository.PostgresSQLProposalRepository
    .lock_for_execution`` - the race-safe backstop this maps to a 409, not a
    generic 400, since the request itself was well-formed."""

    def __init__(self, code: str) -> None:
        super().__init__(f"SQL proposal cannot be executed: {code}")
        self.code = code


class SQLGenerationFailedError(AppError):
    """Raised when SQL generation exhausts its bounded correction-retry
    budget without producing a policy-valid, EXPLAIN-approved query."""

    def __init__(self, last_error_code: str) -> None:
        super().__init__(f"SQL generation failed: {last_error_code}")
        self.last_error_code = last_error_code


class SQLFeatureDisabledError(AppError):
    """Raised when a SQL route/approval endpoint is hit while the feature is
    disabled, not yet rolled out to this principal, or attempted by a
    non-admin during this admin-only rollout phase."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Text-to-SQL is not available: {reason}")
        self.reason = reason


class HybridRetrievalDisabledError(AppError):
    """Raised when a request asks for a retrieval mode the rollout flag
    (``RAGFeatureSettings.hybrid_search_enabled``) currently disallows.

    An explicit error, not a silent fallback to dense: the flag is meant
    to be a real kill switch during the Qdrant-native hybrid rollout, and
    silently downgrading a caller's explicit request would make that gate
    unverifiable from the outside.
    """

    def __init__(self, requested_mode: str) -> None:
        super().__init__(
            f"retrieval_mode={requested_mode!r} is currently disabled by "
            "the server (hybrid retrieval rollout gate)"
        )
        self.requested_mode = requested_mode
