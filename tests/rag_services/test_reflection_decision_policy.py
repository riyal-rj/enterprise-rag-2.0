"""Tests for ThresholdReflectionDecisionPolicy."""

from __future__ import annotations

from app.rag_services.reflection.reflection import (
    ReflectionAction,
    ReflectionBudget,
    ReflectionCritique,
    ReflectionState,
    SupportLevel,
)
from app.rag_services.reflection.reflection_critic import ThresholdReflectionDecisionPolicy


def _policy(**overrides: object) -> ThresholdReflectionDecisionPolicy:
    defaults: dict[str, object] = dict(
        min_evidence_relevance=0.70,
        min_answer_relevance=0.85,
        min_citation_completeness=0.90,
        min_utility=4,
    )
    defaults.update(overrides)
    return ThresholdReflectionDecisionPolicy(**defaults)  # type: ignore[arg-type]


def _critique(**overrides: object) -> ReflectionCritique:
    defaults: dict[str, object] = dict(
        retrieval_needed=False,
        retrieval_query=None,
        evidence_relevance=0.9,
        support_level=SupportLevel.FULL,
        answer_relevance=0.9,
        citation_completeness=0.95,
        utility=5,
        missing_aspects=(),
        unsupported_claims=(),
    )
    defaults.update(overrides)
    return ReflectionCritique(**defaults)  # type: ignore[arg-type]


def _state(**overrides: object) -> ReflectionState:
    defaults: dict[str, object] = dict(answer="answer", evidence=(), iteration=0)
    defaults.update(overrides)
    return ReflectionState(**defaults)  # type: ignore[arg-type]


def _budget(**overrides: object) -> ReflectionBudget:
    defaults: dict[str, object] = dict(
        max_iterations=2,
        max_additional_retrievals=1,
        max_total_tokens=6_000,
        deadline_monotonic=1e12,
    )
    defaults.update(overrides)
    return ReflectionBudget(**defaults)  # type: ignore[arg-type]


def test_fully_supported_high_quality_answer_is_accepted() -> None:
    policy = _policy()

    action = policy.decide(_critique(), _state(), _budget(), deterministic_unsupported_claims=())

    assert action is ReflectionAction.ACCEPT


def test_deterministic_unsupported_claims_block_accept_even_if_critic_says_full_support() -> None:
    """Defense-in-depth: the critic's LLM judgment alone must never be
    sufficient to accept - a deterministic lexical check catching an
    unsupported claim overrides it."""
    policy = _policy()

    action = policy.decide(
        _critique(support_level=SupportLevel.FULL, unsupported_claims=()),
        _state(),
        _budget(),
        deterministic_unsupported_claims=("Customers cannot ever receive transactions.",),
    )

    assert action is not ReflectionAction.ACCEPT


def test_missing_aspects_prevent_accept() -> None:
    policy = _policy()

    action = policy.decide(
        _critique(missing_aspects=("deadline for filing",)),
        _state(),
        _budget(),
        deterministic_unsupported_claims=(),
    )

    assert action is not ReflectionAction.ACCEPT


def test_evidence_relevance_below_floor_retrieves_more_when_budget_allows() -> None:
    policy = _policy()

    action = policy.decide(
        _critique(evidence_relevance=0.4, support_level=SupportLevel.PARTIAL),
        _state(iteration=0, additional_retrievals=0),
        _budget(max_iterations=2, max_additional_retrievals=1),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.RETRIEVE_MORE


def test_retrieval_budget_exhausted_falls_back_to_revise_when_evidence_usable() -> None:
    """evidence_relevance clears the usable floor but support is only
    PARTIAL (so quality_pass still fails) - with the retrieval budget
    already spent, the policy must fall back to REVISE rather than
    ABSTAIN, since the existing evidence is good enough to work with."""
    policy = _policy()

    action = policy.decide(
        _critique(evidence_relevance=0.75, support_level=SupportLevel.PARTIAL),
        _state(iteration=0, additional_retrievals=1),  # retrieval budget already spent
        _budget(max_iterations=2, max_additional_retrievals=1),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.REVISE


def test_no_evidence_at_all_and_no_retrieval_budget_abstains() -> None:
    policy = _policy()

    action = policy.decide(
        _critique(evidence_relevance=0.0, support_level=SupportLevel.NONE),
        _state(iteration=0, additional_retrievals=1),
        _budget(max_iterations=2, max_additional_retrievals=1),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.ABSTAIN


def test_iteration_budget_exhausted_abstains_instead_of_revising_forever() -> None:
    policy = _policy()

    action = policy.decide(
        _critique(citation_completeness=0.5),  # quality_pass fails
        _state(iteration=2),  # == max_iterations
        _budget(max_iterations=2, max_additional_retrievals=1),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.ABSTAIN


def test_usable_evidence_with_revision_budget_revises_rather_than_abstains() -> None:
    policy = _policy()

    action = policy.decide(
        _critique(answer_relevance=0.5),  # quality_pass fails on answer_relevance alone
        _state(iteration=0),
        _budget(max_iterations=2, max_additional_retrievals=1),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.REVISE


def test_exact_threshold_boundary_is_inclusive() -> None:
    policy = _policy(
        min_evidence_relevance=0.70, min_answer_relevance=0.85, min_citation_completeness=0.90
    )

    action = policy.decide(
        _critique(evidence_relevance=0.70, answer_relevance=0.85, citation_completeness=0.90),
        _state(),
        _budget(),
        deterministic_unsupported_claims=(),
    )

    assert action is ReflectionAction.ACCEPT


def test_just_below_threshold_is_not_accepted() -> None:
    policy = _policy(min_answer_relevance=0.85)

    action = policy.decide(
        _critique(answer_relevance=0.849999),
        _state(),
        _budget(),
        deterministic_unsupported_claims=(),
    )

    assert action is not ReflectionAction.ACCEPT


def test_utility_below_floor_prevents_accept() -> None:
    policy = _policy(min_utility=4)

    action = policy.decide(
        _critique(utility=3),
        _state(),
        _budget(),
        deterministic_unsupported_claims=(),
    )

    assert action is not ReflectionAction.ACCEPT
