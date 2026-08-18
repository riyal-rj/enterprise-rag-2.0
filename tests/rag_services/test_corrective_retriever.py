"""Tests for ProductionCorrectiveRetriever and FailSafeCorrectiveRetriever."""

from __future__ import annotations

from app.models.retrieved_chunk import RetrievedChunk
from app.rag_services.crag.corrective_retriever import ProductionCorrectiveRetriever
from app.rag_services.crag.crag import (
    CRAGDecision,
    EvidenceChunk,
    EvidenceOrigin,
    FailSafeCorrectiveRetriever,
    RetrievalGrade,
    WebEvidence,
)


class _FakeGrader:
    """Returns queued ``RetrievalGrade`` results in order - each ``.grade()``
    call pops the next one, so a test can control both the initial grade
    and the post-web-correction re-grade independently."""

    def __init__(self, results: list[RetrievalGrade]) -> None:
        self._results = list(results)
        self.calls: list[list[RetrievedChunk]] = []

    @property
    def cache_namespace(self) -> str:
        return "grader:fake:v1"

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> RetrievalGrade:
        self.calls.append(list(chunks))
        return self._results.pop(0)


class _FakeRefiner:
    def __init__(
        self,
        *,
        local_evidence: tuple[EvidenceChunk, ...] = (),
        web_evidence: tuple[EvidenceChunk, ...] = (),
    ) -> None:
        self._local_evidence = local_evidence
        self._web_evidence = web_evidence
        self.refine_local_calls = 0
        self.refine_web_calls = 0

    @property
    def cache_namespace(self) -> str:
        return "refiner:fake:v1"

    def refine_local(self, question, chunks, grade):  # type: ignore[no-untyped-def]
        self.refine_local_calls += 1
        return self._local_evidence, 1

    def refine_web(self, question, results):  # type: ignore[no-untyped-def]
        self.refine_web_calls += 1
        return self._web_evidence, 1


class _FakeScopePolicy:
    def __init__(self, *, permits: bool = True) -> None:
        self._permits = permits

    @property
    def cache_namespace(self) -> str:
        return "scope:fake:v1"

    def permits_public_regulatory_web(self, question: str) -> bool:
        return self._permits


class _FakeWebRetriever:
    def __init__(
        self, *, results: list[WebEvidence] | None = None, raise_error: bool = False
    ) -> None:
        self._results = results or []
        self._raise_error = raise_error
        self.calls: list[str] = []

    @property
    def cache_namespace(self) -> str:
        return "web:fake:v1"

    def search(self, query: str) -> list[WebEvidence]:
        self.calls.append(query)
        if self._raise_error:
            raise RuntimeError("tavily unavailable")
        return list(self._results)


def _local_evidence(n: int = 1) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(
            text=f"local {i}",
            source="a.pdf",
            page_number=1,
            retrieval_score=0.9,
            origin=EvidenceOrigin.POLICY,
        )
        for i in range(n)
    )


def _web_evidence(n: int = 1) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(
            text=f"web {i}",
            source="RBI Circular",
            page_number=None,
            retrieval_score=0.8,
            origin=EvidenceOrigin.REGULATORY_WEB,
            canonical_url="https://rbi.org.in/x",
            retrieved_at_iso="2026-01-01T00:00:00+00:00",
        )
        for i in range(n)
    )


def _grade(decision: CRAGDecision, coverage: float = 0.9) -> RetrievalGrade:
    return RetrievalGrade(decision=decision, coverage=coverage, chunks=())


def test_correct_local_evidence_returns_without_touching_web() -> None:
    grader = _FakeGrader([_grade(CRAGDecision.CORRECT)])
    refiner = _FakeRefiner(local_evidence=_local_evidence(2))
    web = _FakeWebRetriever(
        results=[
            WebEvidence(
                title="t",
                text="text",
                canonical_url="https://rbi.org.in/x",
                domain="rbi.org.in",
                retrieved_at_iso="2026-01-01T00:00:00+00:00",
                score=0.5,
            )
        ]
    )
    retriever = ProductionCorrectiveRetriever(
        grader=grader, refiner=refiner, scope_policy=_FakeScopePolicy(), web_retriever=web
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.applied is True
    assert outcome.abstain is False
    assert outcome.web_used is False
    assert outcome.evidence == refiner._local_evidence
    assert web.calls == []  # never touched - CORRECT with enough local evidence stops early


def test_ambiguous_with_sufficient_local_evidence_answers_without_web() -> None:
    """Regression: AMBIGUOUS means the retrieved set doesn't *fully* cover
    every part of the question (coverage < correct_threshold) - it does not
    mean there's no usable evidence. A multi-part question answered by one
    genuinely relevant, grader-approved document (e.g. a re-KYC policy's
    debit-restriction clause, with nothing retrieved about credits) must
    still answer from that evidence instead of discarding it and abstaining
    just because the question wasn't completely covered. This was the CRAG
    abstention bug: correct retrieval + correct grading was thrown away."""
    grader = _FakeGrader([_grade(CRAGDecision.AMBIGUOUS, coverage=0.6)])
    refiner = _FakeRefiner(local_evidence=_local_evidence(1))
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=False),
        web_retriever=None,
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.abstain is False
    assert outcome.applied is True
    assert outcome.web_used is False
    assert outcome.evidence == refiner._local_evidence


def test_ambiguous_with_insufficient_local_evidence_abstains_when_web_not_permitted() -> None:
    """Contrast with the test above: when the refiner genuinely extracted
    nothing usable (not just "coverage wasn't complete"), and web correction
    isn't permitted, abstention is still correct - there's no evidence to
    answer from at all."""
    grader = _FakeGrader([_grade(CRAGDecision.AMBIGUOUS, coverage=0.6)])
    refiner = _FakeRefiner(local_evidence=())
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=False),
        web_retriever=None,
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.abstain is True
    assert outcome.bypass_reason == "external_correction_not_permitted"
    assert outcome.web_used is False


def test_incorrect_internal_question_abstains_when_web_not_permitted() -> None:
    grader = _FakeGrader([_grade(CRAGDecision.INCORRECT, coverage=0.1)])
    refiner = _FakeRefiner(local_evidence=())
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=False),
        web_retriever=None,
    )

    outcome = retriever.correct("q", [], allow_web=True)

    assert outcome.abstain is True
    assert outcome.bypass_reason == "external_correction_not_permitted"


def test_empty_web_result_abstains_without_a_second_grade_call() -> None:
    grader = _FakeGrader([_grade(CRAGDecision.AMBIGUOUS, coverage=0.6)])
    # Empty local evidence, not _local_evidence(1): sufficient local
    # evidence now short-circuits before the web branch is ever reached
    # (see test_ambiguous_with_sufficient_local_evidence_answers_without_web) -
    # this test needs the web path to actually run.
    refiner = _FakeRefiner(local_evidence=(), web_evidence=())
    web = _FakeWebRetriever(results=[])
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.abstain is True
    assert outcome.web_used is False
    assert outcome.bypass_reason == "web_correction_empty"
    assert outcome.evidence == refiner._local_evidence
    assert len(grader.calls) == 1  # no re-grade - nothing new to verify


def test_non_correct_regrade_of_combined_evidence_abstains() -> None:
    """Regression: combined local+web evidence being merely non-empty must
    not be treated as "resolved" - the post-correction re-grade must itself
    say CORRECT."""
    grader = _FakeGrader(
        [
            _grade(CRAGDecision.AMBIGUOUS, coverage=0.6),
            _grade(CRAGDecision.AMBIGUOUS, coverage=0.65),
        ]
    )
    # Empty local evidence: sufficient local evidence now short-circuits
    # before the web branch is ever reached (see
    # test_ambiguous_with_sufficient_local_evidence_answers_without_web) -
    # this test needs the web-correction path to actually run.
    refiner = _FakeRefiner(local_evidence=(), web_evidence=_web_evidence(1))
    web = _FakeWebRetriever(
        results=[
            WebEvidence(
                title="t",
                text="text",
                canonical_url="https://rbi.org.in/x",
                domain="rbi.org.in",
                retrieved_at_iso="2026-01-01T00:00:00+00:00",
                score=0.5,
            )
        ]
    )
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.web_used is True
    assert outcome.abstain is True
    assert outcome.bypass_reason == "corrected_evidence_insufficient"
    assert (
        len(outcome.evidence) == 1
    )  # combined local (empty) + web (1) is still returned, just not served as resolved
    assert len(grader.calls) == 2


def test_correct_regrade_below_min_evidence_still_abstains() -> None:
    grader = _FakeGrader(
        [_grade(CRAGDecision.AMBIGUOUS, coverage=0.6), _grade(CRAGDecision.CORRECT, coverage=0.9)]
    )
    refiner = _FakeRefiner(local_evidence=(), web_evidence=_web_evidence(1))
    web = _FakeWebRetriever(
        results=[
            WebEvidence(
                title="t",
                text="text",
                canonical_url="https://rbi.org.in/x",
                domain="rbi.org.in",
                retrieved_at_iso="2026-01-01T00:00:00+00:00",
                score=0.5,
            )
        ]
    )
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
        min_evidence_chunks=3,
    )

    outcome = retriever.correct("q", [], allow_web=True)

    assert outcome.abstain is True
    assert outcome.bypass_reason == "corrected_evidence_insufficient"


def test_verified_corrected_evidence_may_answer() -> None:
    grader = _FakeGrader(
        [_grade(CRAGDecision.AMBIGUOUS, coverage=0.6), _grade(CRAGDecision.CORRECT, coverage=0.9)]
    )
    # Empty local evidence: sufficient local evidence now short-circuits
    # before the web branch is ever reached (see
    # test_ambiguous_with_sufficient_local_evidence_answers_without_web) -
    # this test needs the web-correction/re-grade path to actually run.
    refiner = _FakeRefiner(local_evidence=(), web_evidence=_web_evidence(1))
    web = _FakeWebRetriever(
        results=[
            WebEvidence(
                title="t",
                text="text",
                canonical_url="https://rbi.org.in/x",
                domain="rbi.org.in",
                retrieved_at_iso="2026-01-01T00:00:00+00:00",
                score=0.5,
            )
        ]
    )
    retriever = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
    )

    outcome = retriever.correct(
        "q", [RetrievedChunk(text="x", source="a.pdf", score=0.9)], allow_web=True
    )

    assert outcome.abstain is False
    assert outcome.web_used is True
    assert outcome.bypass_reason is None
    assert len(outcome.evidence) == 1  # local (empty) + web (1)


def test_web_provider_error_returns_fail_safe_fallback_with_local_chunks() -> None:
    grader = _FakeGrader([_grade(CRAGDecision.AMBIGUOUS, coverage=0.6)])
    # Empty local evidence: sufficient local evidence now short-circuits
    # before the web branch (where the raising web retriever lives) is ever
    # reached (see test_ambiguous_with_sufficient_local_evidence_answers_without_web).
    refiner = _FakeRefiner(local_evidence=())
    web = _FakeWebRetriever(raise_error=True)
    delegate = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
    )
    safe = FailSafeCorrectiveRetriever(delegate)
    chunks = [RetrievedChunk(text="x", source="a.pdf", score=0.9)]

    outcome = safe.correct("q", chunks, allow_web=True)

    assert outcome.fallback is True
    assert outcome.abstain is False  # local chunks exist - safe to serve them
    assert outcome.applied is False
    assert len(outcome.evidence) == 1
    assert outcome.evidence[0].text == "x"


def test_fallback_with_no_local_chunks_abstains() -> None:
    grader = _FakeGrader([_grade(CRAGDecision.INCORRECT, coverage=0.0)])
    refiner = _FakeRefiner()
    web = _FakeWebRetriever(raise_error=True)
    delegate = ProductionCorrectiveRetriever(
        grader=grader,
        refiner=refiner,
        scope_policy=_FakeScopePolicy(permits=True),
        web_retriever=web,
    )
    safe = FailSafeCorrectiveRetriever(delegate)

    outcome = safe.correct("q", [], allow_web=True)

    assert outcome.fallback is True
    assert outcome.abstain is True
    assert outcome.evidence == ()


def test_grader_error_is_caught_by_the_fail_safe_wrapper() -> None:
    class _RaisingGrader:
        @property
        def cache_namespace(self) -> str:
            return "grader:raising:v1"

        def grade(self, question, chunks):  # type: ignore[no-untyped-def]
            raise TimeoutError("grader timed out")

    delegate = ProductionCorrectiveRetriever(
        grader=_RaisingGrader(),
        refiner=_FakeRefiner(),
        scope_policy=_FakeScopePolicy(),
        web_retriever=None,
    )
    safe = FailSafeCorrectiveRetriever(delegate)
    chunks = [RetrievedChunk(text="x", source="a.pdf", score=0.9)]

    outcome = safe.correct("q", chunks, allow_web=True)

    assert outcome.fallback is True
    assert outcome.abstain is False
    assert outcome.evidence[0].text == "x"
