"""Bounded whole-answer self-reflection state machine.

Enforces iteration, additional-retrieval, token, and wall-clock budgets.
Budget exhaustion returns a deterministic abstention rather than raising -
a budget limit is a normal policy boundary in a banking deployment, not an
infrastructure fault (unlike an LLM/schema/retrieval error, which the outer
``FailSafeSelfReflectionEngine`` treats as a fallback to the untouched
initial answer instead). Never calls back into ``RAGService.answer()`` -
additional evidence comes from the bounded ``EvidenceAugmenter`` the caller
supplies.
"""

from __future__ import annotations

import time
from dataclasses import replace

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.claim_checker import find_unsupported_claims
from app.rag_services.crag import EvidenceChunk
from app.rag_services.reflection.reflection import (
    EvidenceAugmenter,
    GroundedAnswerReviser,
    ReflectionAction,
    ReflectionBudget,
    ReflectionCritic,
    ReflectionCritique,
    ReflectionDecisionPolicy,
    ReflectionState,
    SelfReflectionOutcome,
    validate_reflection_query,
)

_ABSTENTION = (
    "The supplied approved evidence is insufficient to produce a fully "
    "supported answer to this question."
)


def _dedupe_evidence(
    current: tuple[EvidenceChunk, ...],
    additional: tuple[EvidenceChunk, ...],
    *,
    max_chunks: int,
) -> tuple[EvidenceChunk, ...]:
    result: list[EvidenceChunk] = []
    seen: set[tuple[str, int | None, str]] = set()
    for item in (*current, *additional):
        key = (item.source, item.page_number, " ".join(item.text.casefold().split()))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= max_chunks:
            break
    return tuple(result)


class StructuredSelfReflectionEngine:
    def __init__(
        self,
        *,
        critic: ReflectionCritic,
        policy: ReflectionDecisionPolicy,
        reviser: GroundedAnswerReviser,
        max_iterations: int,
        max_additional_retrievals: int,
        max_total_tokens: int,
        total_timeout_seconds: float,
        max_evidence_chunks: int,
    ) -> None:
        self._critic = critic
        self._policy = policy
        self._reviser = reviser
        self._max_iterations = max_iterations
        self._max_retrievals = max_additional_retrievals
        self._max_total_tokens = max_total_tokens
        self._total_timeout = total_timeout_seconds
        self._max_evidence = max_evidence_chunks

    @property
    def cache_namespace(self) -> str:
        return (
            f"self-reflection:v1:{self._critic.cache_namespace}:"
            f"{self._policy.cache_namespace}:{self._reviser.cache_namespace}:"
            f"iterations={self._max_iterations}:retrievals={self._max_retrievals}:"
            f"tokens={self._max_total_tokens}:evidence={self._max_evidence}:"
            f"deadline={self._total_timeout:.1f}"
        )

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
        *,
        allow_retrieval: bool = True,
    ) -> SelfReflectionOutcome:
        started = time.perf_counter()
        deadline = started + self._total_timeout
        state = ReflectionState(answer=initial_answer, evidence=evidence)
        # The revision-only-canary gate: with allow_retrieval=False (control/
        # shadow/disabled cohorts, or treatment before an admin separately
        # promotes retrieval - see SelfReflectionPlan.allow_retrieval), the
        # effective retrieval budget is 0 regardless of this engine's own
        # configured ceiling, so RETRIEVE_MORE always degrades to REVISE
        # below - the policy never even needs to know about this gate.
        effective_max_retrievals = self._max_retrievals if allow_retrieval else 0

        while True:
            if self._deadline_exceeded(deadline):
                return self._abstain_budget(state, started, reason="budget_exhausted_deadline")
            if state.total_tokens > self._max_total_tokens:
                return self._abstain_budget(state, started, reason="budget_exhausted_tokens")

            remaining = self._remaining_seconds(deadline)
            if remaining <= 0:
                return self._abstain_budget(state, started, reason="budget_exhausted_deadline")

            critique = self._critic.critique(
                question, state.evidence, state.answer, timeout_seconds=remaining
            )
            state = replace(state, total_tokens=state.total_tokens + critique.usage_tokens)
            # Checked immediately after the call returns, not just at the top
            # of the next loop pass - a call that itself overran its
            # timeout_seconds budget (SDK/network jitter, a provider that
            # doesn't strictly enforce the requested timeout) must not be
            # allowed to reach ACCEPT/REVISE/RETRIEVE_MORE execution below.
            if self._deadline_exceeded(deadline):
                return self._abstain_policy(
                    state, started, critique, reason="budget_exhausted_deadline"
                )
            if state.total_tokens > self._max_total_tokens:
                return self._abstain_policy(
                    state, started, critique, reason="budget_exhausted_tokens"
                )

            deterministic_claims = tuple(
                find_unsupported_claims(
                    state.answer,
                    [
                        RetrievedChunk(
                            text=item.text,
                            source=item.source,
                            score=item.retrieval_score,
                            page_number=item.page_number,
                        )
                        for item in state.evidence
                    ],
                )
            )
            budget = ReflectionBudget(
                max_iterations=self._max_iterations,
                max_additional_retrievals=effective_max_retrievals,
                max_total_tokens=self._max_total_tokens,
                deadline_monotonic=deadline,
            )
            action = self._policy.decide(
                critique,
                state,
                budget,
                deterministic_unsupported_claims=deterministic_claims,
            )

            if action is ReflectionAction.ACCEPT:
                # Re-checked one last time: the critique that led to ACCEPT
                # may itself have taken long enough to blow the deadline (see
                # the post-critique check above's rationale) - an accept
                # decided after the caller's budget already ran out must not
                # be returned as a clean success.
                if self._deadline_exceeded(deadline):
                    return self._abstain_policy(
                        state, started, critique, reason="budget_exhausted_deadline"
                    )
                return SelfReflectionOutcome(
                    answer=state.answer,
                    evidence=state.evidence,
                    applied=True,
                    accepted=True,
                    final_action=action,
                    iterations=state.iteration + 1,
                    additional_retrievals=state.additional_retrievals,
                    support_level=critique.support_level,
                    answer_relevance=critique.answer_relevance,
                    citation_completeness=critique.citation_completeness,
                    utility=critique.utility,
                    usage_tokens=state.total_tokens,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                )

            if action is ReflectionAction.ABSTAIN:
                return self._abstain_policy(state, started, critique, reason=None)

            # Engine-level backstop for REVISE/RETRIEVE_MORE, independent of
            # whatever the decision policy did with ``budget`` - guarantees
            # at most ``max_iterations`` revision rounds ever execute (and at
            # most ``max_additional_retrievals`` retrievals), even if a
            # future/non-compliant policy implementation returns an action
            # that ignores its own budget. Checked *before* spending any
            # further tokens/latency on retrieval or revision, not merely
            # relied upon to be caught on a later loop pass.
            if state.iteration >= self._max_iterations:
                return self._abstain_policy(
                    state, started, critique, reason="budget_exhausted_iterations"
                )
            if (
                action is ReflectionAction.RETRIEVE_MORE
                and state.additional_retrievals >= effective_max_retrievals
            ):
                # Retrieval budget already spent - degrade to a plain revise
                # over the existing evidence rather than erroring or
                # abstaining outright; the revision budget just checked above
                # still allows it.
                action = ReflectionAction.REVISE

            if action is ReflectionAction.RETRIEVE_MORE:
                remaining = self._remaining_seconds(deadline)
                if remaining <= 0:
                    return self._abstain_policy(
                        state, started, critique, reason="budget_exhausted_deadline"
                    )
                query = validate_reflection_query(critique.retrieval_query or question)
                additional = augmenter.retrieve(query)
                state = replace(
                    state,
                    evidence=_dedupe_evidence(
                        state.evidence, additional, max_chunks=self._max_evidence
                    ),
                    additional_retrievals=state.additional_retrievals + 1,
                )
                if self._deadline_exceeded(deadline):
                    return self._abstain_policy(
                        state, started, critique, reason="budget_exhausted_deadline"
                    )

            remaining = self._remaining_seconds(deadline)
            if remaining <= 0:
                return self._abstain_policy(
                    state, started, critique, reason="budget_exhausted_deadline"
                )
            revised, revision_tokens = self._reviser.revise(
                question, state.evidence, state.answer, critique, timeout_seconds=remaining
            )
            if self._deadline_exceeded(deadline):
                return self._abstain_policy(
                    state, started, critique, reason="budget_exhausted_deadline"
                )
            state = replace(
                state,
                answer=revised,
                iteration=state.iteration + 1,
                total_tokens=state.total_tokens + revision_tokens,
            )

    @staticmethod
    def _deadline_exceeded(deadline: float) -> bool:
        return time.perf_counter() >= deadline

    @staticmethod
    def _remaining_seconds(deadline: float) -> float:
        return deadline - time.perf_counter()

    @staticmethod
    def _abstain_budget(
        state: ReflectionState, started: float, *, reason: str
    ) -> SelfReflectionOutcome:
        """Budget exhausted before this round's critique ran at all - no
        critique-derived scores to report."""
        return SelfReflectionOutcome(
            answer=_ABSTENTION,
            evidence=state.evidence,
            applied=True,
            accepted=False,
            final_action=ReflectionAction.ABSTAIN,
            iterations=state.iteration,
            additional_retrievals=state.additional_retrievals,
            abstain=True,
            bypass_reason=reason,
            usage_tokens=state.total_tokens,
            duration_ms=(time.perf_counter() - started) * 1_000,
        )

    @staticmethod
    def _abstain_policy(
        state: ReflectionState,
        started: float,
        critique: ReflectionCritique,
        *,
        reason: str | None,
    ) -> SelfReflectionOutcome:
        """The policy decided ABSTAIN (or the token budget tipped over right
        after this round's critique ran) - report that critique's scores."""
        return SelfReflectionOutcome(
            answer=_ABSTENTION,
            evidence=state.evidence,
            applied=True,
            accepted=False,
            final_action=ReflectionAction.ABSTAIN,
            iterations=state.iteration + 1,
            additional_retrievals=state.additional_retrievals,
            abstain=True,
            bypass_reason=reason,
            support_level=critique.support_level,
            answer_relevance=critique.answer_relevance,
            citation_completeness=critique.citation_completeness,
            utility=critique.utility,
            usage_tokens=state.total_tokens,
            duration_ms=(time.perf_counter() - started) * 1_000,
        )
