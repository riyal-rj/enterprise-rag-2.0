"""Admin-mutable reranker selection: backend, canary rollout percentage, and
an emergency kill switch, all changeable from the RAG Operations panel
without a process restart (see ``app.controllers.rag_ops_controller``).

``app.api.deps.get_dynamic_reranker`` returns a single long-lived instance
of this class, sharing a ``RagRuntimeConfigStore`` with ``RAGService`` and
``QdrantSemanticQueryCache`` (see ``app.rag_services.rag_runtime_config``);
reading ``backend``/``rollout_percentage``/``emergency_disabled`` from that
store's current snapshot at the top of each call is what lets an admin's
config-update request take effect on the very next chat request, with all
three of this class's own mutable fields guaranteed to come from the same
atomically-applied snapshot instead of being independently, non-atomically
mutable.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.rag_runtime_config import (
    RagRuntimeConfig,
    RagRuntimeConfigStore,
    RerankerBackendName,
)
from app.rag_services.reranker import FailOpenReranker, NoOpReranker, ReRanker, ReRankOutcome
from app.services.rag_metrics_service import RagMetricsService

logger = logging.getLogger(__name__)


class DynamicReranker:
    """:class:`ReRanker` that reads its backend/rollout/kill-switch state
    from a shared, atomically-swapped config snapshot instead of frozen
    settings - see module docstring."""

    def __init__(
        self,
        *,
        local: ReRanker,
        voyage: ReRanker | None,
        metrics: RagMetricsService,
        config_store: RagRuntimeConfigStore,
    ) -> None:
        self._local = local
        self._voyage = voyage
        self._metrics = metrics
        self._config_store = config_store

    @property
    def _config(self) -> RagRuntimeConfig:
        return self._config_store.current

    @property
    def has_voyage_backend(self) -> bool:
        """Whether a Voyage delegate was actually constructed (i.e. an API
        key is configured) - lets the ops controller reject switching to
        ``voyage`` before it ever reaches a real request."""
        return self._voyage is not None

    @property
    def name(self) -> str:
        backend = self._config.reranker_backend
        try:
            return self._delegate(backend).name
        except RuntimeError:
            return backend

    @property
    def cache_namespace(self) -> str:
        # One snapshot for this whole property, so backend/rollout/
        # emergency-disabled - all folded in below so a cache entry
        # produced under one rollout config can't be served back after an
        # admin changes it, same reasoning as RAGService._cache_key's
        # candidate-pool-size versioning - always describe the same config,
        # never a torn mix of an old and a new field.
        config = self._config
        try:
            delegate_namespace = self._delegate(config.reranker_backend).cache_namespace
        except RuntimeError:
            delegate_namespace = f"unconfigured:{config.reranker_backend}"
        return (
            f"dynamic:v1:backend={config.reranker_backend}"
            f":rollout={config.reranker_rollout_percentage}"
            f":emergency={int(config.reranker_emergency_disabled)}"
            f":{delegate_namespace}"
        )

    def rerank(
        self, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        # Captured once, up front: every field this call consults below
        # (emergency-disabled, rollout%, backend) must come from the same
        # snapshot, or a config update landing mid-call could apply its new
        # backend while still honoring its old emergency-disabled state.
        config = self._config

        if config.reranker_emergency_disabled:
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)

        if not _sampled_in(query, config.reranker_rollout_percentage):
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)

        try:
            delegate = self._delegate(config.reranker_backend)
        except RuntimeError:
            logger.exception(
                "rag_ops.reranker_misconfigured", extra={"backend": config.reranker_backend}
            )
            self._metrics.record_rerank(duration_ms=0.0, fallback=True, usage_tokens=None)
            noop_outcome = NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)
            return ReRankOutcome(
                items=noop_outcome.items,
                backend=config.reranker_backend,
                applied=False,
                fallback=True,
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

    def _delegate(self, backend: RerankerBackendName) -> ReRanker:
        if backend == "voyage":
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
