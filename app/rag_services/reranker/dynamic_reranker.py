"""Admin-mutable reranker selection: backend, canary rollout percentage, and
an emergency kill switch, all changeable from the RAG Operations panel
without a process restart (see ``app.controllers.rag_ops_controller``).

``DynamicReranker`` holds no mutable state of its own - ``plan()`` decides a
query's cohort (disabled/control/treatment) from a single, caller-supplied
``RagRuntimeConfig`` snapshot (``RAGService.answer`` captures exactly one
per request and passes it in), and ``execute()`` performs the reranking
according to a previously-computed plan. Nothing here re-reads a shared
store mid-request, which is what makes it safe for ``RAGService`` to build
a cohort-aware cache namespace from the same plan it later executes
against - see ``app.rag_services.rag_runtime_config`` and
``app.rag_services.rag_service`` for why a second, independent config read
partway through a request would be a real bug (a concurrent admin update
could split one request across two cohorts).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RerankerBackendName
from app.rag_services.reranker.reranker import (
    FailOpenReranker,
    NoOpReranker,
    PlannedReranker,
    ReRanker,
    ReRankOutcome,
    RerankPlan,
)
from app.rag_services.rollout import sampled_in
from app.services.rag_metrics_service import RagMetricsService

logger = logging.getLogger(__name__)


class DynamicReranker(PlannedReranker):
    """Production :class:`PlannedReranker`: cohort decided from a
    caller-supplied config snapshot - see module docstring."""

    def __init__(
        self,
        *,
        local: ReRanker,
        voyage: ReRanker | None,
        metrics: RagMetricsService,
    ) -> None:
        self._local = local
        self._voyage = voyage
        self._metrics = metrics

    @property
    def has_voyage_backend(self) -> bool:
        """Whether a Voyage delegate was actually constructed (i.e. an API
        key is configured) - lets the ops controller reject switching to
        ``voyage`` before it ever reaches a real request."""
        return self._voyage is not None

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> RerankPlan:
        if not enabled:
            return RerankPlan("disabled", None, "none", "disabled", "rerank:none:reason=disabled")
        if config.emergency_disabled:
            return RerankPlan(
                "disabled",
                None,
                "none",
                "emergency_disabled",
                "rerank:none:reason=emergency_disabled",
            )
        if not sampled_in(query, config.reranker_rollout_percentage, salt="reranker:v1"):
            return RerankPlan("control", None, "none", "rollout", "rerank:none:reason=rollout")

        backend = config.reranker_backend
        try:
            delegate_namespace = self._delegate(backend).cache_namespace
            reported_backend = self._delegate(backend).name
        except RuntimeError:
            delegate_namespace = f"unconfigured:{backend}"
            reported_backend = backend
        return RerankPlan(
            "treatment",
            backend,
            reported_backend,
            None,
            f"dynamic:v1:cohort=treatment:backend={backend}:{delegate_namespace}",
        )

    def execute(
        self, plan: RerankPlan, *, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> ReRankOutcome:
        if plan.cohort != "treatment":
            return NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)

        assert plan.backend_key is not None  # guaranteed by plan() for cohort == "treatment"
        try:
            delegate = self._delegate(plan.backend_key)
        except RuntimeError:
            logger.exception("rag_ops.reranker_misconfigured", extra={"backend": plan.backend_key})
            self._metrics.record_rerank(duration_ms=0.0, fallback=True, usage_tokens=None)
            noop_outcome = NoOpReranker().rerank(query=query, candidates=candidates, top_k=top_k)
            return ReRankOutcome(
                items=noop_outcome.items,
                backend=plan.backend_key,
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
