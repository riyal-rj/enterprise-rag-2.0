"""CLI: diff two eval report JSON files.

python -m app.eval.diff eval/results/..._naive.json eval/results/..._all.json
"""

from __future__ import annotations

import argparse
import json
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


def diff_reports(before: EvalPayload, after: EvalPayload) -> None:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two eval report JSON files.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    diff_reports(_load_payload(args.before), _load_payload(args.after))


if __name__ == "__main__":
    main()
