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
            f"tokens={self._max_total_tokens}:evidence={self._max_evidence}"
        )

    def reflect(
        self,
        question: str,
        evidence: tuple[EvidenceChunk, ...],
        initial_answer: str,
        augmenter: EvidenceAugmenter,
    ) -> SelfReflectionOutcome:
        started = time.perf_counter()
        deadline = started + self._total_timeout
        state = ReflectionState(answer=initial_answer, evidence=evidence)

        while True:
            if time.perf_counter() >= deadline:
                return self._abstain_budget(state, started, reason="budget_exhausted_deadline")
            if state.total_tokens > self._max_total_tokens:
                return self._abstain_budget(state, started, reason="budget_exhausted_tokens")
            # Defense-in-depth iteration cap, independent of whatever the
            # decision policy itself does with ``budget.max_iterations`` -
            # guarantees this loop terminates after at most
            # ``max_iterations + 1`` critique calls even if a future/custom
            # policy implementation doesn't respect the budget it's handed.
            if state.iteration > self._max_iterations:
                return self._abstain_budget(state, started, reason="budget_exhausted_iterations")

            critique = self._critic.critique(question, state.evidence, state.answer)
            state = replace(state, total_tokens=state.total_tokens + critique.usage_tokens)
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
                max_additional_retrievals=self._max_retrievals,
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

            current_evidence = state.evidence
            retrievals = state.additional_retrievals
            if action is ReflectionAction.RETRIEVE_MORE:
                query = validate_reflection_query(critique.retrieval_query or question)
                additional = augmenter.retrieve(query)
                current_evidence = _dedupe_evidence(
                    current_evidence, additional, max_chunks=self._max_evidence
                )
                retrievals += 1

            revised, revision_tokens = self._reviser.revise(
                question, current_evidence, state.answer, critique
            )
            state = ReflectionState(
                answer=revised,
                evidence=current_evidence,
                iteration=state.iteration + 1,
                additional_retrievals=retrievals,
                total_tokens=state.total_tokens + revision_tokens,
            )

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
