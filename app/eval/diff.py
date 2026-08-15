"""CLI: diff two eval report JSON files.

python -m app.eval.diff eval/results/..._naive.json eval/results/..._all.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from app.eval.ragas_adapter import METRIC_NAMES as RAGAS_METRIC_NAMES
from app.eval.types import EvalPayload, EvalRow


def _load_payload(path: Path) -> EvalPayload:
    """Load a report JSON file, trusting it was produced by ``run_ragas``."""
    return cast(EvalPayload, json.loads(path.read_text(encoding="utf-8")))


def _row_ok(row: EvalRow) -> bool:
    forbidden_check = row.get("forbidden_check")
    source_overlap = row.get("source_overlap")
    if not isinstance(forbidden_check, dict) or not isinstance(source_overlap, dict):
        return False
    return bool(forbidden_check.get("passed")) and bool(source_overlap.get("passed"))


def _attempted_case_ids(payload: EvalPayload) -> set[str]:
    """Every case id the run attempted, whether it produced a scored row
    or an intentional skip. Deliberately excludes ``errors`` (see
    ``run_ragas._invoke_all``) - a genuine invocation failure isn't a
    case this profile chose not to run."""
    row_ids = {row["id"] for row in payload.get("rows", [])}
    skipped_ids = {entry["id"] for entry in payload.get("skipped", [])}
    return row_ids | skipped_ids


def check_case_parity(before: EvalPayload, after: EvalPayload) -> bool:
    """Warn (or, for the same-cases-but-different-outcome case, flag as a
    real problem) when ``before``/``after`` don't cover the same cases -
    the improved/regressed comparison below silently drops any case id
    absent from either side, so a mismatch here would otherwise pass
    unnoticed.

    Returns ``False`` when both reports attempted the exact same cases
    but scored a different subset of them (``rows`` differ) - that
    specifically means a case succeeded under one profile and
    errored/was skipped under the other despite an identical case set,
    which is a correctness signal worth failing the comparison over, not
    just a heads-up. A difference in the *attempted* sets (e.g. one
    report used ``--filter`` and the other didn't) is only a heads-up -
    comparing a filtered run against an unfiltered one is a legitimate
    workflow, not a bug.
    """
    before_attempted, after_attempted = _attempted_case_ids(before), _attempted_case_ids(after)
    if before_attempted != after_attempted:
        only_before = sorted(before_attempted - after_attempted)
        only_after = sorted(after_attempted - before_attempted)
        print(
            "\nWARNING: the two reports attempted different case sets "
            "(different --filter?) - the comparison below only covers the overlap."
        )
        if only_before:
            print(f"  only attempted in before: {only_before}")
        if only_after:
            print(f"  only attempted in after: {only_after}")
        return True

    before_rows = {row["id"] for row in before.get("rows", [])}
    after_rows = {row["id"] for row in after.get("rows", [])}
    if before_rows != after_rows:
        print(
            "\nWARNING: both reports attempted the same cases, but scored a different "
            "subset - some case(s) errored or were skipped on one side only, silently "
            "dropped from the improved/regressed comparison below:"
        )
        print(f"  scored only in before: {sorted(before_rows - after_rows)}")
        print(f"  scored only in after: {sorted(after_rows - before_rows)}")
        return False

    return True


def diff_reports(before: EvalPayload, after: EvalPayload) -> bool:
    """Print the comparison; returns whether the two reports had matching
    case-id parity (see :func:`check_case_parity`) - ``main()`` uses this
    to fail the run when they don't."""
    parity_ok = check_case_parity(before, after)

    before_agg, after_agg = before["aggregate"], after["aggregate"]
    print(f"{before['profile']}: forbidden_violations={before_agg.get('forbidden_violations', 0)}")
    print(f"{after['profile']}: forbidden_violations={after_agg.get('forbidden_violations', 0)}")

    print("\nRAGAS metric deltas (after - before):")
    for name in RAGAS_METRIC_NAMES:
        b, a = before_agg.get(name), after_agg.get(name)
        if b is None or a is None:
            continue
        print(f"  {name:<20} {b:.3f} -> {a:.3f}  ({a - b:+.3f})")

    print("\nRetrieval metric deltas (after - before):")
    for name in ("hit_rate", "mrr"):
        b, a = before_agg.get(name), after_agg.get(name)
        if b is None or a is None:
            continue
        print(f"  {name:<20} {b:.3f} -> {a:.3f}  ({a - b:+.3f})")

    before_by_id = {row["id"]: row for row in before.get("rows", [])}
    after_by_id = {row["id"]: row for row in after.get("rows", [])}

    improved: list[str] = []
    regressed: list[str] = []
    for case_id, after_row in after_by_id.items():
        before_row = before_by_id.get(case_id)
        if before_row is None:
            continue
        before_ok, after_ok = _row_ok(before_row), _row_ok(after_row)
        if not before_ok and after_ok:
            improved.append(case_id)
        elif before_ok and not after_ok:
            regressed.append(case_id)

    print(f"\nImproved ({len(improved)}): {', '.join(sorted(improved)) or 'none'}")
    print(f"Regressed ({len(regressed)}): {', '.join(sorted(regressed)) or 'none'}")

    return parity_ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two eval report JSON files.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parity_ok = diff_reports(_load_payload(args.before), _load_payload(args.after))
    if not parity_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
