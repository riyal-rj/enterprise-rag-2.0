"""Query-transformation contracts and safe decorators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTransformOutcome:
    """Result of preparing texts for dense retrieval.

    ``retrieval_texts`` are never evidence and must never be sent to the
    answer LLM. When ``applied`` is false, the tuple contains the original
    question so the caller can safely embed it.
    """

    retrieval_texts: tuple[str, ...]
    backend: str
    applied: bool
    fallback: bool = False
    bypass_reason: str | None = None
    usage_tokens: int = 0
    duration_ms: float = 0.0


class QueryTransformer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def cache_namespace(self) -> str: ...

    def transform(self, query: str) -> QueryTransformOutcome: ...


class NoOpQueryTransformer:
    def __init__(self, *, reason: str = "disabled") -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "none"

    @property
    def cache_namespace(self) -> str:
        return f"query-transform:none:reason={self._reason}"

    def transform(self, query: str) -> QueryTransformOutcome:
        return QueryTransformOutcome(
            retrieval_texts=(query,),
            backend=self.name,
            applied=False,
            bypass_reason=self._reason,
        )


class FailOpenQueryTransformer:
    """Convert any transformer failure into original-query retrieval."""

    def __init__(self, delegate: QueryTransformer) -> None:
        self._delegate = delegate

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def cache_namespace(self) -> str:
        return self._delegate.cache_namespace

    def transform(self, query: str) -> QueryTransformOutcome:
        try:
            return self._delegate.transform(query)
        except Exception as exc:  # noqa: BLE001 - deliberate availability boundary
            logger.warning(
                "rag.hyde_fallback",
                extra={"backend": self._delegate.name, "error_type": type(exc).__name__},
            )
            return QueryTransformOutcome(
                retrieval_texts=(query,),
                backend=self._delegate.name,
                applied=False,
                fallback=True,
                bypass_reason="transform_error",
            )
