"""Self-reflective RAG package: contracts, critic, reviser, engine, rollout."""

from __future__ import annotations

from app.rag_services.reflection.answer_reviser import (
    StructuredGroundedAnswerReviser,
    find_fabricated_citations,
)
from app.rag_services.reflection.dynamic_self_reflection import (
    DynamicSelfReflectionEngine,
    StaticPlannedSelfReflectionEngine,
)
from app.rag_services.reflection.reflection import (
    EvidenceAugmenter,
    FailSafeSelfReflectionEngine,
    GroundedAnswerReviser,
    PlannedNoOpSelfReflectionEngine,
    PlannedSelfReflectionEngine,
    ReflectionAction,
    ReflectionBudget,
    ReflectionCritic,
    ReflectionCritique,
    ReflectionDecisionPolicy,
    ReflectionState,
    SelfReflectionCohort,
    SelfReflectionEngine,
    SelfReflectionOutcome,
    SelfReflectionPlan,
    SupportLevel,
    validate_reflection_query,
)
from app.rag_services.reflection.reflection_critic import (
    StructuredReflectionCritic,
    ThresholdReflectionDecisionPolicy,
)
from app.rag_services.reflection.reflection_engine import StructuredSelfReflectionEngine
from app.rag_services.reflection.telemetry import (
    NoOpSelfReflectionTelemetry,
    SelfReflectionTelemetry,
)

__all__ = [
    "DynamicSelfReflectionEngine",
    "EvidenceAugmenter",
    "FailSafeSelfReflectionEngine",
    "GroundedAnswerReviser",
    "NoOpSelfReflectionTelemetry",
    "PlannedNoOpSelfReflectionEngine",
    "PlannedSelfReflectionEngine",
    "ReflectionAction",
    "ReflectionBudget",
    "ReflectionCritic",
    "ReflectionCritique",
    "ReflectionDecisionPolicy",
    "ReflectionState",
    "SelfReflectionCohort",
    "SelfReflectionEngine",
    "SelfReflectionOutcome",
    "SelfReflectionPlan",
    "SelfReflectionTelemetry",
    "StaticPlannedSelfReflectionEngine",
    "StructuredGroundedAnswerReviser",
    "StructuredReflectionCritic",
    "StructuredSelfReflectionEngine",
    "SupportLevel",
    "ThresholdReflectionDecisionPolicy",
    "find_fabricated_citations",
    "validate_reflection_query",
]
