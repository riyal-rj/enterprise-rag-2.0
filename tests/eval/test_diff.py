from __future__ import annotations

from typing import cast

from app.eval.diff import check_case_parity, diff_reports
from app.eval.types import EvalPayload


def _payload(
    rows: list[str] | None = None,
    skipped: list[str] | None = None,
    forbidden_violations: int = 0,
) -> EvalPayload:
    return cast(
        EvalPayload,
        {
            "profile": "p",
            "flags": {},
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "filter": None,
            "mode": "service",
            "rows": [
                {
                    "id": case_id,
                    "demonstrates_feature": "baseline",
                    "intent": "rag",
                    "question": "q",
                    "answer": "a",
                    "contexts": [],
                    "ground_truth": "",
                    "actual_sources": [],
                    "ranked_sources": [],
                    "golden_sources": [],
                    "forbidden_keywords": [],
                    "forbidden_check": {"passed": True, "found": []},
                    "source_overlap": {"passed": True, "ratio": 1.0, "matched": []},
                }
                for case_id in (rows or [])
            ],
            "skipped": [{"id": case_id, "reason": "skip"} for case_id in (skipped or [])],
            "errors": [],
            "aggregate": {
                "faithfulness": None,
                "context_precision": None,
                "context_recall": None,
                "answer_relevancy": None,
                "forbidden_violations": forbidden_violations,
                "hit_rate": None,
                "mrr": None,
            },
        },
    )


def test_check_case_parity_true_when_both_reports_cover_the_same_cases() -> None:
    before = _payload(rows=["q-1", "q-2"])
    after = _payload(rows=["q-1", "q-2"])

    assert check_case_parity(before, after) is True


def test_check_case_parity_true_but_warns_when_attempted_sets_differ(capsys) -> None:
    """A filtered vs. unfiltered comparison is a legitimate workflow, not
    a bug - it's a warning, not a failure."""
    before = _payload(rows=["q-1", "q-2"])
    after = _payload(rows=["q-1"])

    assert check_case_parity(before, after) is True
    assert "different case sets" in capsys.readouterr().out


def test_check_case_parity_false_when_same_cases_attempted_but_different_rows_scored(
    capsys,
) -> None:
    """Same attempted set (q-1 succeeded in both, q-2 attempted in both)
    but q-2 only scored in `before` - it errored or was skipped in
    `after`. That's a real correctness signal, not just a filter
    difference, so this must return False."""
    before = _payload(rows=["q-1", "q-2"])
    after = _payload(rows=["q-1"], skipped=["q-2"])

    assert check_case_parity(before, after) is False
    assert "scored a different subset" in capsys.readouterr().out


def test_diff_reports_returns_the_parity_result() -> None:
    matching = _payload(rows=["q-1"])
    mismatched_but_same_attempted = _payload(rows=[], skipped=["q-1"])

    assert diff_reports(matching, matching) is True
    assert diff_reports(matching, mismatched_but_same_attempted) is False
