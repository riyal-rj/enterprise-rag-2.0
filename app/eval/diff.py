"""CLI: diff two eval report JSON files.

    python -m app.eval.diff eval/results/..._naive.json eval/results/..._all.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.eval.runner import EvalReport


def _load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def diff_reports(before: EvalReport, after: EvalReport) -> None:
    before_by_id = {result.case_id: result for result in before.results}
    after_by_id = {result.case_id: result for result in after.results}

    print(f"{before.profile}: {before.passed}/{before.total} ({before.pass_rate:.0%})")
    print(f"{after.profile}: {after.passed}/{after.total} ({after.pass_rate:.0%})")

    improved: list[str] = []
    regressed: list[str] = []
    for case_id, after_result in after_by_id.items():
        before_result = before_by_id.get(case_id)
        if before_result is None:
            continue
        if not before_result.passed and after_result.passed:
            improved.append(case_id)
        elif before_result.passed and not after_result.passed:
            regressed.append(case_id)

    print(f"\nImproved ({len(improved)}): {', '.join(sorted(improved)) or 'none'}")
    print(f"Regressed ({len(regressed)}): {', '.join(sorted(regressed)) or 'none'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two eval report JSON files.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    diff_reports(_load_report(args.before), _load_report(args.after))


if __name__ == "__main__":
    main()
