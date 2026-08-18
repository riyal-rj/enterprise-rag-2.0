"""End-to-end eval-pipeline tests for CRAG: ServiceInvoker -> run_ragas._invoke_all/_build_row.

Complements tests/eval/test_invokers.py (unit tests for ServiceInvoker's
integrity checks) and tests/eval/test_run_ragas.py (unit tests for the
skip/error/fail-on-skip exit-code logic) by proving the two compose
correctly for CRAG specifically: a requested-but-broken CRAG case becomes an
*error* bucket entry (not silently skipped), and a row that does reach
scoring carries exactly CRAG's final evidence as its RAGAS context.
"""

from __future__ import annotations

from typing import cast

from app.eval import run_ragas
from app.eval.invokers import ServiceInvoker
from app.eval.profiles import PROFILES, PipelineProfile
from app.eval.schemas import EvalFeature, ExpectedOutcome, GoldenCase, Intent
from app.rag_services.rag_service import RAGService
from app.schemas.chat import (
    ChatResponse,
    CRAGMetadata,
    EvidencePreview,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    RetrievedChunkPreview,
)


def _case(case_id: str, feature: EvalFeature = EvalFeature.CRAG) -> GoldenCase:
    return GoldenCase(
        id=case_id,
        question=f"question for {case_id}",
        intent=Intent.RAG,
        golden_sources=["doc.pdf"],
        golden_answer_keywords=["keyword"],
        demonstrates_feature=feature,
        expected_baseline=ExpectedOutcome.PASS,
        expected_with_feature="pass",
        notes="test fixture",
    )


class _FakeRAGService:
    def __init__(self, response: ChatResponse) -> None:
        self._response = response

    def answer(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str | None = None,
        *,
        reranking_enabled: bool | None = None,
        hyde_enabled: bool | None = None,
        crag_enabled: bool | None = None,
    ) -> ChatResponse:
        return self._response


def _response(
    *,
    crag_applied: bool = True,
    crag_fallback: bool = False,
    final_evidence: list[EvidencePreview] | None = None,
) -> ChatResponse:
    return ChatResponse(
        answer="the answer",
        sources=["doc.pdf"],
        confidence=0.9,
        metadata=ResponseMetadata(
            route="rag",
            retrieval_mode="hybrid",
            hyde=HyDEMetadata(enabled=False, backend="none"),
            reranking=RerankingMetadata(enabled=True, applied=True, backend="fake"),
            crag=CRAGMetadata(
                enabled=True, applied=crag_applied, fallback=crag_fallback, decision="correct"
            ),
            retrieved_chunks=[
                RetrievedChunkPreview(
                    text="raw chunk", source="doc.pdf", score=0.9, page_number=None
                )
            ],
            final_evidence=final_evidence if final_evidence is not None else [],
        ),
    )


def _profile() -> PipelineProfile:
    return PROFILES["hybrid+rerank+crag"]


def test_requested_crag_fallback_becomes_an_error_not_a_skip() -> None:
    fake_service = _FakeRAGService(_response(crag_applied=False, crag_fallback=True))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_service))
    cases = [_case("q-901")]

    rows, skipped, errors = run_ragas._invoke_all(invoker, cases, _profile())

    assert rows == []
    assert skipped == []
    assert len(errors) == 1
    assert errors[0]["id"] == "q-901"


def test_requested_crag_with_no_final_evidence_becomes_an_error() -> None:
    fake_service = _FakeRAGService(_response(crag_applied=True, final_evidence=[]))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_service))
    cases = [_case("q-902")]

    rows, skipped, errors = run_ragas._invoke_all(invoker, cases, _profile())

    assert rows == []
    assert len(errors) == 1
    assert "no final-evidence" in errors[0]["reason"]


def test_ragas_context_uses_final_evidence_not_retrieved_chunks() -> None:
    final_evidence = [
        EvidencePreview(text="corrected sentence one", source="a.pdf", score=0.9, origin="policy"),
        EvidencePreview(
            text="corrected sentence two",
            source="RBI Circular",
            score=0.8,
            origin="regulatory_web",
            canonical_url="https://rbi.org.in/x",
        ),
    ]
    fake_service = _FakeRAGService(_response(final_evidence=final_evidence))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_service))
    cases = [_case("q-903")]

    rows, skipped, errors = run_ragas._invoke_all(invoker, cases, _profile())

    assert errors == []
    assert skipped == []
    assert rows[0]["contexts"] == ["corrected sentence one", "corrected sentence two"]
    # The canonical URL, not the web item's title, is what reaches ranked_sources.
    assert rows[0]["ranked_sources"] == ["a.pdf", "https://rbi.org.in/x"]


def test_successful_crag_run_has_zero_skipped_and_a_scored_row(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    final_evidence = [EvidencePreview(text="evidence", source="a.pdf", score=0.9, origin="policy")]
    fake_service = _FakeRAGService(_response(final_evidence=final_evidence))
    invoker = ServiceInvoker(rag_service=cast(RAGService, fake_service))
    cases = [_case("q-904")]

    rows, skipped, errors = run_ragas._invoke_all(invoker, cases, _profile())
    monkeypatch.setattr(run_ragas, "score_with_ragas", lambda rows: [{} for _ in rows])
    run_ragas._score_rows(rows)

    assert len(rows) == 1
    assert skipped == []
    assert errors == []


def test_zero_scored_rows_fails_the_run(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import sys

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv", ["run_ragas", "--profile", "hybrid+rerank+crag", "--output", str(out_path)]
    )
    fake_service = _FakeRAGService(_response(crag_applied=False, crag_fallback=True))
    monkeypatch.setattr(
        run_ragas,
        "_select_invoker",
        lambda mode: ServiceInvoker(rag_service=cast(RAGService, fake_service)),
    )
    monkeypatch.setattr(run_ragas, "_load_cases", lambda *args, **kwargs: [_case("q-905")])

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        run_ragas.main()

    assert exc_info.value.code == 1
