"""Tests for StructuredSelfReflectionEngine's bounded state machine."""

from __future__ import annotations

import time

from app.rag_services.crag import EvidenceChunk, EvidenceOrigin
from app.rag_services.reflection.reflection import (
    ReflectionAction,
    ReflectionCritique,
    SupportLevel,
)
from app.rag_services.reflection.reflection_engine import StructuredSelfReflectionEngine


def _evidence(text: str = "text", source: str = "a.pdf") -> EvidenceChunk:
    return EvidenceChunk(
        text=text, source=source, page_number=1, retrieval_score=0.9, origin=EvidenceOrigin.POLICY
    )


def _critique(**overrides: object) -> ReflectionCritique:
    defaults: dict[str, object] = dict(
        retrieval_needed=False,
        retrieval_query=None,
        evidence_relevance=0.9,
        support_level=SupportLevel.FULL,
        answer_relevance=0.9,
        citation_completeness=0.9,
        utility=5,
        missing_aspects=(),
        unsupported_claims=(),
        usage_tokens=10,
    )
    defaults.update(overrides)
    return ReflectionCritique(**defaults)  # type: ignore[arg-type]


class _FakeCritic:
    def __init__(self, critiques: list[ReflectionCritique], *, sleep_seconds: float = 0.0) -> None:
        self._critiques = critiques
        self._sleep_seconds = sleep_seconds
        self.calls = 0
        self.received_timeouts: list[float | None] = []

    @property
    def cache_namespace(self) -> str:
        return "critic:fake:v1"

    def critique(
        self,
        question: str,
        evidence: object,
        answer: str,
        *,
        timeout_seconds: float | None = None,
    ) -> ReflectionCritique:
        self.received_timeouts.append(timeout_seconds)
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        result = self._critiques[self.calls]
        self.calls += 1
        return result


class _FakePolicy:
    def __init__(self, actions: list[ReflectionAction]) -> None:
        self._actions = actions
        self.calls = 0

    @property
    def cache_namespace(self) -> str:
        return "policy:fake:v1"

    def decide(
        self,
        critique: object,
        state: object,
        budget: object,
        *,
        deterministic_unsupported_claims: object,
    ) -> ReflectionAction:
        result = self._actions[self.calls]
        self.calls += 1
        return result


class _FakeReviser:
    def __init__(self, answers: list[str], tokens: int = 5, *, sleep_seconds: float = 0.0) -> None:
        self._answers = answers
        self._tokens = tokens
        self._sleep_seconds = sleep_seconds
        self.calls = 0

    @property
    def cache_namespace(self) -> str:
        return "reviser:fake:v1"

    def revise(
        self,
        question: str,
        evidence: object,
        previous_answer: str,
        critique: object,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[str, int]:
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        result = self._answers[self.calls]
        self.calls += 1
        return result, self._tokens


class _FakeAugmenter:
    def __init__(self, evidence: tuple[EvidenceChunk, ...], *, sleep_seconds: float = 0.0) -> None:
        self._evidence = evidence
        self._sleep_seconds = sleep_seconds
        self.queries: list[str] = []

    def retrieve(self, query: str) -> tuple[EvidenceChunk, ...]:
        self.queries.append(query)
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        return self._evidence


def _engine(
    *, critic: object, policy: object, reviser: object, **overrides: object
) -> StructuredSelfReflectionEngine:
    defaults: dict[str, object] = dict(
        critic=critic,
        policy=policy,
        reviser=reviser,
        max_iterations=2,
        max_additional_retrievals=1,
        max_total_tokens=6_000,
        total_timeout_seconds=25.0,
        max_evidence_chunks=10,
    )
    defaults.update(overrides)
    return StructuredSelfReflectionEngine(**defaults)  # type: ignore[arg-type]


def test_first_pass_accept_uses_exactly_one_critic_call() -> None:
    critic = _FakeCritic([_critique()])
    policy = _FakePolicy([ReflectionAction.ACCEPT])
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.accepted is True
    assert outcome.final_action is ReflectionAction.ACCEPT
    assert outcome.answer == "initial answer"
    assert outcome.iterations == 1
    assert critic.calls == 1
    assert reviser.calls == 0


def test_revise_then_accept() -> None:
    critic = _FakeCritic([_critique(answer_relevance=0.5), _critique()])
    policy = _FakePolicy([ReflectionAction.REVISE, ReflectionAction.ACCEPT])
    reviser = _FakeReviser(["revised answer"])
    engine = _engine(critic=critic, policy=policy, reviser=reviser)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.accepted is True
    assert outcome.answer == "revised answer"
    assert outcome.iterations == 2
    assert critic.calls == 2
    assert reviser.calls == 1


def test_retrieve_then_revise_then_accept() -> None:
    extra_evidence = (_evidence(text="new evidence", source="b.pdf"),)
    critic = _FakeCritic(
        [
            _critique(
                evidence_relevance=0.2,
                support_level=SupportLevel.NONE,
                retrieval_needed=True,
                retrieval_query="missing KYC policy",
            ),
            _critique(),
        ]
    )
    policy = _FakePolicy([ReflectionAction.RETRIEVE_MORE, ReflectionAction.ACCEPT])
    reviser = _FakeReviser(["revised with new evidence"])
    augmenter = _FakeAugmenter(extra_evidence)
    engine = _engine(critic=critic, policy=policy, reviser=reviser)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", augmenter)

    assert outcome.additional_retrievals == 1
    assert augmenter.queries == ["missing KYC policy"]
    assert extra_evidence[0] in outcome.evidence
    assert outcome.answer == "revised with new evidence"
    assert reviser.calls == 1


def test_evidence_is_deduped_and_capped_after_retrieval() -> None:
    duplicate = _evidence(text="same text", source="a.pdf")
    original = (duplicate,)
    additional = (duplicate, _evidence(text="unique", source="c.pdf"))
    critic = _FakeCritic(
        [
            _critique(
                retrieval_needed=True,
                retrieval_query="more evidence",
                evidence_relevance=0.2,
                support_level=SupportLevel.PARTIAL,
            ),
            _critique(),
        ]
    )
    policy = _FakePolicy([ReflectionAction.RETRIEVE_MORE, ReflectionAction.ACCEPT])
    reviser = _FakeReviser(["revised"])
    augmenter = _FakeAugmenter(additional)
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_evidence_chunks=10)

    outcome = engine.reflect("q", original, "initial answer", augmenter)

    texts = {item.text for item in outcome.evidence}
    assert texts == {"same text", "unique"}  # the exact duplicate collapsed to one


def test_policy_abstain_returns_deterministic_abstention_text() -> None:
    critic = _FakeCritic([_critique(support_level=SupportLevel.NONE)])
    policy = _FakePolicy([ReflectionAction.ABSTAIN])
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.accepted is False
    assert outcome.final_action is ReflectionAction.ABSTAIN
    assert "insufficient" in outcome.answer
    assert outcome.support_level is SupportLevel.NONE  # critique scores still reported


def test_token_budget_exhaustion_abstains_without_calling_policy() -> None:
    critic = _FakeCritic([_critique(usage_tokens=1_000)])
    policy = _FakePolicy([ReflectionAction.ACCEPT])  # would accept if ever called
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_total_tokens=500)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_tokens"
    assert policy.calls == 0
    assert critic.calls == 1


def test_revise_is_skipped_when_token_budget_already_exhausted_by_critique() -> None:
    """Regression: the token budget was previously only checked before
    critique and right after it returns - never immediately before the
    reviser call - so a revise could still be issued (and spend further
    tokens/latency) even though the budget was already exhausted."""
    critic = _FakeCritic([_critique(usage_tokens=100)])
    policy = _FakePolicy([ReflectionAction.REVISE])
    reviser = _FakeReviser(["should never be used"])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_total_tokens=100)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_tokens"
    assert reviser.calls == 0


def test_deadline_exhaustion_abstains_without_calling_the_critic() -> None:
    critic = _FakeCritic([_critique()])
    policy = _FakePolicy([ReflectionAction.ACCEPT])
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, total_timeout_seconds=0.0)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_deadline"
    assert critic.calls == 0


def test_iteration_cap_terminates_even_if_the_policy_never_stops() -> None:
    """Defense-in-depth: the engine itself must bound iterations, not rely
    solely on a policy correctly respecting its budget. The backstop fires
    *before* executing a REVISE/RETRIEVE_MORE the policy requested past
    budget, not just on a later loop pass - so with max_iterations=2, only
    2 revisions ever actually execute, even though a 3rd critique still
    happens (to see whether the 2nd revision is now acceptable)."""
    critic = _FakeCritic([_critique(answer_relevance=0.1)] * 10)
    policy = _FakePolicy([ReflectionAction.REVISE] * 10)  # never accepts/abstains on its own
    reviser = _FakeReviser(["revised"] * 10)
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_iterations=2)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_iterations"
    assert critic.calls == 3
    assert reviser.calls == 2


def test_zero_max_iterations_never_executes_a_single_revision() -> None:
    """Regression for the exact off-by-one a non-compliant policy could
    exploit: with max_iterations=0, a policy that returns REVISE anyway must
    never reach the reviser - not even once. The old top-of-loop-only check
    (`state.iteration > max_iterations`) let exactly one revision slip
    through here, since it only blocked the *next* loop pass, not the
    REVISE this pass already decided on."""
    critic = _FakeCritic([_critique(answer_relevance=0.1)])
    policy = _FakePolicy([ReflectionAction.REVISE])  # non-compliant: ignores budget.max_iterations
    reviser = _FakeReviser(["should never be returned"])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_iterations=0)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_iterations"
    assert outcome.answer != "should never be returned"
    assert reviser.calls == 0


def test_retrieve_more_past_retrieval_budget_degrades_to_revise() -> None:
    """A non-compliant policy that keeps returning RETRIEVE_MORE after the
    retrieval budget is spent must not call the augmenter again - the engine
    degrades the action to a plain REVISE over the existing evidence
    instead (still gated by the iteration backstop above)."""
    critic = _FakeCritic(
        [
            _critique(retrieval_needed=True, retrieval_query="q1", evidence_relevance=0.2),
            _critique(),
        ]
    )
    policy = _FakePolicy([ReflectionAction.RETRIEVE_MORE, ReflectionAction.ACCEPT])
    reviser = _FakeReviser(["revised without a second retrieval"])
    augmenter = _FakeAugmenter((_evidence(text="extra", source="b.pdf"),))
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_additional_retrievals=0)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", augmenter)

    assert augmenter.queries == []  # retrieval budget was already 0 - never called
    assert outcome.answer == "revised without a second retrieval"
    assert outcome.additional_retrievals == 0


# ---- Deadline enforcement around every collaborator call, not just at the
# top of the loop - regression coverage for a real repro: a critic call
# that itself takes longer than the total deadline must never let the loop
# reach ACCEPT, because the only check used to happen before the call, not
# after it returned. ----


def test_slow_critic_call_that_overruns_the_deadline_is_not_accepted() -> None:
    critic = _FakeCritic([_critique()], sleep_seconds=0.05)
    policy = _FakePolicy([ReflectionAction.ACCEPT])
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, total_timeout_seconds=0.01)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.accepted is False
    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_deadline"


def test_critic_receives_the_remaining_deadline_not_the_full_stage_timeout() -> None:
    critic = _FakeCritic([_critique()])
    policy = _FakePolicy([ReflectionAction.ACCEPT])
    reviser = _FakeReviser([])
    engine = _engine(critic=critic, policy=policy, reviser=reviser, total_timeout_seconds=5.0)

    engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert critic.received_timeouts[0] is not None
    assert critic.received_timeouts[0] <= 5.0


def test_slow_reviser_call_that_overruns_the_deadline_aborts() -> None:
    critic = _FakeCritic([_critique(answer_relevance=0.1)])
    policy = _FakePolicy([ReflectionAction.REVISE])
    reviser = _FakeReviser(["should never be reported as the final answer"], sleep_seconds=0.05)
    engine = _engine(critic=critic, policy=policy, reviser=reviser, total_timeout_seconds=0.02)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_deadline"
    assert outcome.answer != "should never be reported as the final answer"


def test_slow_augmenter_retrieve_that_overruns_the_deadline_aborts() -> None:
    critic = _FakeCritic(
        [
            _critique(
                retrieval_needed=True,
                retrieval_query="missing evidence query",
                evidence_relevance=0.2,
            )
        ]
    )
    policy = _FakePolicy([ReflectionAction.RETRIEVE_MORE])
    reviser = _FakeReviser(["should never be reached"])
    augmenter = _FakeAugmenter((_evidence(text="extra", source="b.pdf"),), sleep_seconds=0.05)
    engine = _engine(critic=critic, policy=policy, reviser=reviser, total_timeout_seconds=0.02)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", augmenter)

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_deadline"
    assert reviser.calls == 0  # never reached the reviser after the slow retrieval


def test_critic_and_reviser_disagreement_loop_still_terminates_bounded() -> None:
    """Uses the *real* ThresholdReflectionDecisionPolicy (not a fake that
    unconditionally returns REVISE) driven by a critic that cites a
    genuinely different reason to reject the answer on every round - never
    the same complaint twice, so nothing about the engine or policy can
    special-case "the same issue keeps recurring." Round 1: missing an
    aspect. Round 2 (after the reviser "fixes" that): now an unsupported
    claim. Round 3: now citation completeness is short. The critic and
    reviser never converge - proves the engine's own iteration backstop
    terminates this regardless of how many distinct failure modes a
    disagreeing critic/reviser pair cycles through."""
    from app.rag_services.reflection.reflection_critic import ThresholdReflectionDecisionPolicy

    real_policy = ThresholdReflectionDecisionPolicy(
        min_evidence_relevance=0.70,
        min_answer_relevance=0.85,
        min_citation_completeness=0.90,
        min_utility=4,
    )
    critiques = [
        _critique(missing_aspects=("filing deadline",)),
        _critique(support_level=SupportLevel.PARTIAL, unsupported_claims=("a wild claim",)),
        _critique(citation_completeness=0.5),
        _critique(answer_relevance=0.5),
        _critique(utility=2),
    ]
    critic = _FakeCritic(critiques * 3)  # far more rounds available than any budget allows
    reviser = _FakeReviser([f"revision {i}" for i in range(15)])
    engine = _engine(critic=critic, policy=real_policy, reviser=reviser, max_iterations=3)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    # A real, budget-respecting policy recognizes its own revision budget is
    # exhausted and returns ABSTAIN itself (bypass_reason=None - a policy
    # decision, not the engine's defense-in-depth backstop, which only ever
    # fires against a non-compliant policy - see
    # test_zero_max_iterations_never_executes_a_single_revision for that
    # case) - either way, the loop is bounded regardless of how many
    # distinct disagreements the critic raises.
    assert outcome.bypass_reason is None
    # max_iterations=3 permits exactly 3 revise rounds (4 critiques total:
    # the initial one plus one per revision) before the policy itself abstains.
    assert critic.calls == 4
    assert reviser.calls == 3
