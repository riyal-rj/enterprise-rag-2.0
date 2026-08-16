"""Admin-mutable reranker selection: backend, canary rollout percentage, and
an emergency kill switch, all changeable from the RAG Operations panel
without a process restart (see ``app.controllers.rag_ops_controller``).

``app.api.deps.get_dynamic_reranker`` returns a single long-lived instance
of this class; mutating it in place (``set_backend`` /
``set_rollout_percentage`` / ``set_emergency_disabled``) is what lets an
admin's config-update request take effect on the very next chat request,
since ``RAGService`` always holds this same object rather than a config
snapshot taken at construction time.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from typing import Literal

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.reranker import FailOpenReranker, NoOpReranker, ReRanker, ReRankOutcome
from app.services.rag_metrics_service import RagMetricsService

logger = logging.getLogger(__name__)

RerankerBackendName = Literal["local", "voyage"]


class DynamicReranker:
    """:class:`ReRanker` that reads its backend/rollout/kill-switch state
    from mutable instance fields instead of frozen settings - see module
    docstring."""

    def __init__(
        self,
        *,
        local: ReRanker,
        voyage: ReRanker | None,
        metrics: RagMetricsService,
        backend: RerankerBackendName = "local",
        rollout_percentage: int = 100,
        emergency_disabled: bool = False,
    ) -> None:
        self._local = local
        self._voyage = voyage
        self._metrics = metrics
        self._backend: RerankerBackendName = backend
        self._rollout_percentage = rollout_percentage
        self._emergency_disabled = emergency_disabled

    def set_backend(self, backend: RerankerBackendName) -> None:
        self._backend = backend

    def set_rollout_percentage(self, percentage: int) -> None:
        if not 0 <= percentage <= 100:
            raise ValueError("percentage must be between 0 and 100")
        self._rollout_percentage = percentage

    def set_emergency_disabled(self, disabled: bool) -> None:
        self._emergency_disabled = disabled

    @property
    def has_voyage_backend(self) -> bool:
        """Whether a Voyage delegate was actually constructed (i.e. an API
        key is configured) - lets the ops controller reject switching to
        ``voyage`` before it ever reaches a real request."""
        return self._voyage is not None

    @property
    def name(self) -> str:
        try:
            return self._delegate().name
        except RuntimeError:
            return self._backend

    @property
    def cache_namespace(self) -> str:
        # Rollout% and emergency-disabled are folded in alongside the
        # backend so a cache entry produced under one rollout config can't
        # be served back after an admin changes it - same reasoning as
        # RAGService._cache_key's candidate-pool-size versioning.
        try:
            delegate_namespace = self._delegate().cache_namespace
        except RuntimeError:
            delegate_namespace = f"unconfigured:{self._backend}"
        return (
            f"dynamic:v1:backend={self._backend}"
            f":rollout={self._rollout_percentage}"
            f":emergency={int(self._emergency_disabled)}"
            f":{delegate_namespace}"
        )

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        if self._emergency_disabled:
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)

        if not _sampled_in(query, self._rollout_percentage):
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)

        try:
            delegate = self._delegate()
        except RuntimeError:
            logger.exception("rag_ops.reranker_misconfigured", extra={"backend": self._backend})
            self._metrics.record_rerank(duration_ms=0.0, fallback=True, usage_tokens=None)
            noop_outcome = NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)
            return ReRankOutcome(
                items=noop_outcome.items, backend=self._backend, applied=False, fallback=True
            )

        started = time.perf_counter()
        outcome = FailOpenReranker(delegate=delegate, fallback=NoOpReranker()).rerank(
            query=query, candidates=candidates, top_k=top_k
        )
        duration_ms = (time.perf_counter() - started) * 1000
        self._metrics.record_rerank(
            duration_ms=duration_ms, fallback=outcome.fallback, usage_tokens=outcome.usage_tokens
        )
        return outcome

    def _delegate(self) -> ReRanker:
        if self._backend == "voyage":
            if self._voyage is None:
                raise RuntimeError("reranker_backend=voyage but no Voyage API key is configured")
            return self._voyage
        return self._local


def _sampled_in(query: str, rollout_percentage: int) -> bool:
    """Deterministic per-question rollout sample: hash the question text
    rather than a per-call random draw, so the *same* question always gets
    the same treatment.

    A random per-call coin flip would race with caching: RAGService caches
    under a key that doesn't vary per rerank attempt, so whichever outcome
    is computed first for a given question would be served to every later
    ask of it regardless of its own flip - on a small, high-repeat-rate
    policy corpus like this one, that would make "rollout percentage"
    mostly measure "percentage of *first-ever* asks", not sustained
    traffic. Hashing the question makes the split stable and immune to
    that race.
    """
    if rollout_percentage >= 100:
        return True
    if rollout_percentage <= 0:
        return False
    digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    return (int(digest, 16) % 100) < rollout_percentage
