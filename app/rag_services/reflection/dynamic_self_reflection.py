"""Deterministic production rollout for self-reflective RAG."""

from __future__ import annotations

from app.rag_services.crag import EvidenceChunk
from app.rag_services.rag_runtime_config import RagRuntimeConfig
from app.rag_services.reflection.reflection import (
    EvidenceAugmenter,
    FailSafeSelfReflectionEngine,
    PlannedNoOpSelfReflectionEngine,
    PlannedSelfReflectionEngine,
    SelfReflectionEngine,
    SelfReflectionOutcome,
    SelfReflectionPlan,
)
from app.rag_services.rollout import sampled_in


class DynamicSelfReflectionEngine(PlannedSelfReflectionEngine):
    """The reflection stage that actually serves real ``/chat`` traffic -
    admin-mutable (rollout%/shadow/retrieval-gate/emergency-disable) via the
    RAG Operations panel. Holds no config state of its own -
    ``RAGService.answer`` passes in the one ``RagRuntimeConfig`` snapshot it
    captures per request."""

    def __init__(self, *, delegate: SelfReflectionEngine) -> None:
        self._delegate = FailSafeSelfReflectionEngine(delegate)

    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> SelfReflectionPlan:
        if not enabled:
            return SelfReflectionPlan(
                "disabled", "disabled", "self-reflection:none:reason=disabled"
            )
        if config.emergency_disabled:
            return SelfReflectionPlan(
                "disabled",
                "emergency_disabled",
                "self-reflection:none:reason=emergency_disabled",
            )
        if config.self_reflective_shadow_enabled:
            # Observe-only: every request actually runs reflection (for
            # measurement) but RAGService never serves it - see that
            # module's "shadow" cohort handling, mirroring CRAG's shadow
            # mode. Retrieval stays unavailable here regardless of
            # self_reflective_retrieval_enabled, ahead of the rollout-
            # percentage check below, since shadow mode isn't sampled
            # traffic - it's a blanket observability stage.
            return SelfReflectionPlan("shadow", None, "self-reflection:shadow:v1", False)
        if not sampled_in(
            question, config.self_reflective_rollout_percentage, salt="self_reflective:v1"
        ):
            return SelfReflectionPlan("control", "rollout", "self-reflection:none:reason=rollout")
        allow_retrieval = config.self_reflective_retrieval_enabled
        return SelfReflectionPlan(
            "treatment",
            None,
            f"self-reflection:dynamic:v1:cohort=treatment:retrieval={allow_retrieval}:"
            f"{self._delegate.cache_namespace}",
            allow_retrieval,
        )

    def execute(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        plan: SelfReflectionPlan,
    ) -> SelfReflectionOutcome:
        if plan.cohort not in ("treatment", "shadow"):
            return PlannedNoOpSelfReflectionEngine().execute(
                question, evidence, initial_answer, augmenter, plan
            )
        return self._delegate.reflect(
            question, evidence, initial_answer, augmenter, allow_retrieval=plan.allow_retrieval
        )


class StaticPlannedSelfReflectionEngine(PlannedSelfReflectionEngine):
    """Eval adapter: enabled means treatment, independent of RAG Ops admin
    rollout/shadow/retrieval-gate/emergency state - mirrors
    ``StaticPlannedCorrectiveRetriever``. Retrieval is always allowed when
    enabled, so an eval run exercises the complete pipeline regardless of
    what's currently promoted to real traffic."""

    def __init__(self, *, delegate: SelfReflectionEngine) -> None:
        self._delegate = FailSafeSelfReflectionEngine(delegate)

    def plan(self, question: str, config: RagRuntimeConfig, *, enabled: bool) -> SelfReflectionPlan:
        if not enabled:
            return SelfReflectionPlan(
                "disabled", "disabled", "self-reflection:none:reason=disabled"
            )
        return SelfReflectionPlan(
            "treatment",
            None,
            f"self-reflection:static:v1:retrieval=True:{self._delegate.cache_namespace}",
            True,
        )

    def execute(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        plan: SelfReflectionPlan,
    ) -> SelfReflectionOutcome:
        if plan.cohort != "treatment":
            return PlannedNoOpSelfReflectionEngine().execute(
                question, evidence, initial_answer, augmenter, plan
            )
        return self._delegate.reflect(
            question, evidence, initial_answer, augmenter, allow_retrieval=plan.allow_retrieval
        )
