from __future__ import annotations

import logging

import pytest

from app.eval import run_ragas
from app.eval.invokers import InvokeResponse, RetrievedChunk, ServiceInvoker, SkippedIntent
from app.eval.profiles import PROFILES, PipelineProfile
from app.eval.schemas import EvalFeature, ExpectedOutcome, GoldenCase, Intent


def _case(case_id: str, feature: EvalFeature = EvalFeature.BASELINE) -> GoldenCase:
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


class _FakeInvoker:
    """Returns a canned response, or raises, per question."""

    def __init__(self, canned: dict[str, InvokeResponse | Exception]) -> None:
        self._canned = canned

    def invoke(
        self, question: str, flags: PipelineProfile, intent: Intent
    ) -> tuple[InvokeResponse, list[RetrievedChunk]]:
        result = self._canned[question]
        if isinstance(result, Exception):
            raise result
        return result, [RetrievedChunk(text="chunk text", source="doc.pdf")]


def test_select_invoker_service_mode_returns_service_invoker() -> None:
    assert isinstance(run_ragas._select_invoker("service"), ServiceInvoker)


def test_select_invoker_api_mode_raises_not_implemented_not_sys_exit() -> None:
    """Testability check: this must raise, not call sys.exit(), or a test
    hitting this path would kill the test process."""
    with pytest.raises(NotImplementedError, match="api"):
        run_ragas._select_invoker("api")


def test_load_cases_no_filter_returns_all_forty() -> None:
    assert len(run_ragas._load_cases(None, None)) == 40


def test_load_cases_filter_includes_baseline_as_control_set() -> None:
    cases = run_ragas._load_cases(None, "hyde")
    features = {case.demonstrates_feature for case in cases}

    assert features == {EvalFeature.HYDE, EvalFeature.BASELINE}
    assert len(cases) == 9  # 4 baseline + 5 hyde


def test_build_row_shapes_the_row_correctly() -> None:
    case = _case("q-991")
    response = InvokeResponse(answer="the answer", sources=["doc.pdf"])
    chunks = [RetrievedChunk(text="chunk 1", source="doc.pdf")]

    row = run_ragas._build_row(case, response, chunks)

    assert row["id"] == "q-991"
    assert row["answer"] == "the answer"
    assert row["contexts"] == ["chunk 1"]
    assert row["ground_truth"] == "keyword"
    assert "ragas_metrics" not in row  # not scored yet


def test_invoke_all_buckets_skipped_vs_errored_vs_rows() -> None:
    cases = [_case("q-991"), _case("q-992"), _case("q-993")]
    invoker = _FakeInvoker(
        {
            "question for q-991": InvokeResponse(answer="ok", sources=["doc.pdf"]),
            "question for q-992": SkippedIntent("not applicable"),
            "question for q-993": ValueError("boom"),
        }
    )

    rows, skipped = run_ragas._invoke_all(invoker, cases, PROFILES["naive"])

    assert [row["id"] for row in rows] == ["q-991"]
    assert {entry["id"] for entry in skipped} == {"q-992", "q-993"}
    q993 = next(entry for entry in skipped if entry["id"] == "q-993")
    assert "error: boom" in q993["reason"]


def test_score_rows_survives_ragas_failure_without_losing_rows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_scorer(rows: object) -> object:
        raise RuntimeError("ragas is down")

    monkeypatch.setattr(run_ragas, "score_with_ragas", broken_scorer)

    rows = [run_ragas._build_row(_case("q-991"), InvokeResponse(answer="ok"), [])]

    with caplog.at_level(logging.ERROR):
        run_ragas._score_rows(rows)

    assert rows[0]["ragas_metrics"] == {}
    assert rows[0]["forbidden_check"]["passed"] is True
    assert "eval.ragas_scoring_failed" in caplog.text


def test_resolve_output_path_uses_explicit_output_when_given() -> None:
    import datetime

    path = run_ragas._resolve_output_path(
        "custom.json", "naive", datetime.datetime.now(datetime.UTC)
    )

    assert str(path) == "custom.json"


def test_resolve_output_path_defaults_to_timestamped_results_path() -> None:
    import datetime

    timestamp = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
    path = run_ragas._resolve_output_path(None, "naive", timestamp)

    assert str(path).replace("\\", "/") == "eval/results/20260102T030405Z_naive.json"
