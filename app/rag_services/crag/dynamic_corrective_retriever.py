"""Deterministic production rollout for CRAG."""

from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag import (
    CorrectiveRetriever,
    CRAGOutcome,
    CRAGPlan,
    FailSafeCorrectiveRetriever,
    PlannedCorrectiveRetriever,
    PlannedNoOpCorrectiveRetriever,
)
from app.rag_services.rag_runtime_config import RagRuntimeConfig
from app.rag_services.rollout import sampled_in


class DynamicCorrectiveRetriever(PlannedCorrectiveRetriever):
    def __init__(self, *, delegate: CorrectiveRetriever) -> None:
        self._delegate = FailSafeCorrectiveRetriever(delegate)

    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> CRAGPlan:
        if not enabled:
            return CRAGPlan("disabled", "disabled", False, "crag:none:reason=disabled")
        if config.emergency_disabled:
            return CRAGPlan(
                "disabled", "emergency_disabled", False, "crag:none:reason=emergency_disabled"
            )
        if config.crag_shadow_enabled:
            # Observe-only: every request actually runs correction (for
            # measurement) but RAGService never serves it - see that
            # module's "shadow" cohort handling. allow_web is always False
            # here, ahead of the rollout-percentage check below, since
            # shadow mode isn't sampled traffic - it's a blanket
            # observability stage.
            return CRAGPlan("shadow", None, False, "crag:shadow:v1")
        if not sampled_in(question, config.crag_rollout_percentage, salt="crag:v1"):
            return CRAGPlan("control", "rollout", False, "crag:none:reason=rollout")
        return CRAGPlan(
            "treatment",
            None,
            config.crag_web_enabled,
            f"crag:dynamic:v1:cohort=treatment:{self._delegate.cache_namespace}",
        )

    def execute(self, question: str, chunks: list[RetrievedChunk], plan: CRAGPlan) -> CRAGOutcome:
        if plan.cohort not in ("treatment", "shadow"):
            return PlannedNoOpCorrectiveRetriever().execute(question, chunks, plan)
        return self._delegate.correct(question, chunks, allow_web=plan.allow_web)
