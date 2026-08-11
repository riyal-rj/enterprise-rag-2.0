import pytest

from app.eval.reporting import aggregate, print_table


def _row(
    case_id: str,
    feature: str,
    *,
    forbidden_ok: bool = True,
    forbidden_found: list[str] | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    return {
        "id": case_id,
        "demonstrates_feature": feature,
        "forbidden_check": {"passed": forbidden_ok, "found": forbidden_found or []},
        "source_overlap": {"passed": True, "ratio": 1.0, "matched": []},
        "ragas_metrics": metrics or {},
    }


def test_aggregate_empty_rows_has_none_metrics_and_zero_violations() -> None:
    summary = aggregate([])

    assert summary["faithfulness"] is None
    assert summary["context_precision"] is None
    assert summary["context_recall"] is None
    assert summary["answer_relevancy"] is None
    assert summary["forbidden_violations"] == 0


def test_aggregate_computes_rounded_means_and_violation_count() -> None:
    rows = [
        _row("q-001", "baseline", forbidden_ok=True, metrics={"faithfulness": 0.9}),
        _row(
            "q-002",
            "baseline",
            forbidden_ok=False,
            forbidden_found=["kubeconfig"],
            metrics={"faithfulness": 0.7},
        ),
    ]

    summary = aggregate(rows)

    assert summary["faithfulness"] == 0.8
    assert summary["context_precision"] is None  # no row scored it
    assert summary["forbidden_violations"] == 1


def test_print_table_reports_no_rows_evaluated_when_everything_skipped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "profile": "naive",
        "mode": "service",
        "filter": None,
        "rows": [],
        "skipped": [{"id": "q-001", "reason": "service pipeline not wired yet"}],
        "aggregate": aggregate([]),
    }

    print_table(payload)

    captured = capsys.readouterr()
    assert "no rows evaluated" in captured.out


def test_print_table_shows_forbidden_found_keywords_not_a_missing_hits_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = _row("q-033", "wild", forbidden_ok=False, forbidden_found=["kubeconfig"])
    payload = {
        "profile": "all",
        "mode": "service",
        "filter": None,
        "rows": [row],
        "skipped": [],
        "aggregate": aggregate([row]),
    }

    print_table(payload)

    captured = capsys.readouterr()
    assert "FAIL: ['kubeconfig']" in captured.out
