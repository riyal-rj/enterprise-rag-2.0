"""Tests for StructuredSelfReflectionEngine's bounded state machine."""

from __future__ import annotations

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
    def __init__(self, critiques: list[ReflectionCritique]) -> None:
        self._critiques = critiques
        self.calls = 0

    @property
    def cache_namespace(self) -> str:
        return "critic:fake:v1"

    def critique(self, question: str, evidence: object, answer: str) -> ReflectionCritique:
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
    def __init__(self, answers: list[str], tokens: int = 5) -> None:
        self._answers = answers
        self._tokens = tokens
        self.calls = 0

    @property
    def cache_namespace(self) -> str:
        return "reviser:fake:v1"

    def revise(
        self, question: str, evidence: object, previous_answer: str, critique: object
    ) -> tuple[str, int]:
        result = self._answers[self.calls]
        self.calls += 1
        return result, self._tokens


class _FakeAugmenter:
    def __init__(self, evidence: tuple[EvidenceChunk, ...]) -> None:
        self._evidence = evidence
        self.queries: list[str] = []

    def retrieve(self, query: str) -> tuple[EvidenceChunk, ...]:
        self.queries.append(query)
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
    solely on a policy correctly respecting its budget."""
    critic = _FakeCritic([_critique(answer_relevance=0.1)] * 10)
    policy = _FakePolicy([ReflectionAction.REVISE] * 10)  # never accepts/abstains on its own
    reviser = _FakeReviser(["revised"] * 10)
    engine = _engine(critic=critic, policy=policy, reviser=reviser, max_iterations=2)

    outcome = engine.reflect("q", (_evidence(),), "initial answer", _FakeAugmenter(()))

    assert outcome.abstain is True
    assert outcome.bypass_reason == "budget_exhausted_iterations"
    # max_iterations=2 permits iteration 0, 1, 2 to each run a full
    # critique+revise round (3 critic calls); the 4th loop pass sees
    # state.iteration=3 > 2 and abstains before critiquing again.
    assert critic.calls == 3
    assert reviser.calls == 3
