"""Admin-mutable HyDE rollout: canary percentage and an emergency kill
switch, changeable from the RAG Operations panel without a process restart
(see ``app.controllers.rag_ops_controller``).

``DynamicQueryTransformer`` holds no mutable state of its own - ``plan()``
decides a query's cohort from a single, caller-supplied ``RagRuntimeConfig``
snapshot (``RAGService.answer`` captures exactly one per request and passes
it in), and ``execute()`` performs the transformation according to a
previously-computed plan. Nothing here re-reads a shared store mid-request,
which is what makes it safe for ``RAGService`` to build a cohort-aware cache
namespace from the same plan it later executes against - see
``app.rag_services.rag_runtime_config`` and ``app.rag_services.rag_service``
for why a second, independent config read partway through a request would
be a real bug (a concurrent admin update could split one request across two
cohorts).
"""

from __future__ import annotations

from app.rag_services.hyde.query_transformer import (
    FailOpenQueryTransformer,
    HyDEPlan,
    NoOpQueryTransformer,
    PlannedQueryTransformer,
    QueryTransformer,
    QueryTransformOutcome,
)
from app.rag_services.rag_runtime_config import RagRuntimeConfig
from app.rag_services.rollout import sampled_in


class DynamicQueryTransformer(PlannedQueryTransformer):
    """Production :class:`PlannedQueryTransformer`: cohort decided from a
    caller-supplied config snapshot - see module docstring."""

    def __init__(self, *, delegate: QueryTransformer) -> None:
        self._delegate = FailOpenQueryTransformer(delegate)

    def plan(self, query: str, config: RagRuntimeConfig, *, enabled: bool) -> HyDEPlan:
        if not enabled:
            return HyDEPlan("disabled", "disabled", "query-transform:none:reason=disabled")
        if config.emergency_disabled:
            return HyDEPlan(
                "disabled", "emergency_disabled", "query-transform:none:reason=emergency_disabled"
            )
        if not sampled_in(query, config.hyde_rollout_percentage, salt="hyde:v1"):
            return HyDEPlan("control", "rollout", "query-transform:none:reason=rollout")
        return HyDEPlan(
            "treatment",
            None,
            f"dynamic-query-transform:v1:cohort=treatment:{self._delegate.cache_namespace}",
        )

    def execute(self, query: str, plan: HyDEPlan) -> QueryTransformOutcome:
        if plan.cohort != "treatment":
            return NoOpQueryTransformer(reason=plan.bypass_reason or "disabled").transform(query)
        return self._delegate.transform(query)
