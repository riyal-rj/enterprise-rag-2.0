"""Writes eval reports to disk and prints a console summary."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.eval.runner import EvalReport

_DEFAULT_RESULTS_DIR = Path("eval/results")


def write_report(report: EvalReport, results_dir: Path | None = None) -> Path:
    """Write ``report`` as JSON and return the file path.

    Filename ends in ``_{profile}.json`` (``+`` replaced with ``-``), e.g.
    ``20260809_143000_naive.json`` — matches the glob the Makefile's
    ``eval-diff`` target looks for.
    """
    directory = results_dir or _DEFAULT_RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    profile_slug = report.profile.replace("+", "-")
    path = directory / f"{timestamp}_{profile_slug}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_summary(report: EvalReport) -> None:
    print(f"\nProfile: {report.profile}")
    print(f"Overall: {report.passed}/{report.total} passed ({report.pass_rate:.0%})")

    print("\nBy feature:")
    for feature, breakdown in sorted(report.by_feature.items()):
        print(f"  {feature:<18} {breakdown.passed}/{breakdown.total} ({breakdown.pass_rate:.0%})")

    mismatches = report.mismatched_expectations
    if mismatches:
        print(f"\n{len(mismatches)} case(s) didn't match their documented expectation:")
        for result in mismatches:
            outcome = "passed" if result.passed else "failed"
            print(f"  {result.case_id}: {outcome}, expected {result.expected.value}")
