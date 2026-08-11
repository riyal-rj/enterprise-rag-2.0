"""CLI: run golden eval cases against a pipeline profile via RAGAS.

    python -m app.eval.run_ragas --profile naive
    python -m app.eval.run_ragas --profile hybrid+rerank+hyde --filter hyde

See the Makefile's ``eval-*`` targets for the exact profile/filter
combinations used in CI/local workflows.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from app.eval.invokers import ServiceInvoker, SkippedIntent
from app.eval.loader import load_golden_cases
from app.eval.post_checks import forbidden_keywords_check, source_overlap
from app.eval.profiles import PROFILES
from app.eval.ragas_adapter import run as run_ragas
from app.eval.reporting import aggregate, print_table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden eval cases against a pipeline profile.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to the golden cases YAML (default: app/eval/data/goldens.yaml).",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run cases whose demonstrates_feature == FILTER (plus baseline).",
    )
    parser.add_argument("--mode", default="service", choices=["service", "api"], help="Invocation mode")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/<timestamp>_<profile>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    flags = PROFILES[args.profile]
    goldens = load_golden_cases(Path(args.questions) if args.questions else None)

    if args.filter:
        goldens = [
            case for case in goldens if case.demonstrates_feature.value in (args.filter, "baseline")
        ]

    if args.mode == "service":
        invoker = ServiceInvoker()
    else:
        print("API mode not yet implemented (Phase B).", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for case in goldens:
        try:
            resp, chunks = invoker.invoke(case.question, flags, case.intent)
        except SkippedIntent as exc:
            skipped.append({"id": case.id, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - one bad case shouldn't kill the run
            skipped.append({"id": case.id, "reason": f"error: {exc}"})
            continue

        rows.append(
            {
                "id": case.id,
                "demonstrates_feature": case.demonstrates_feature.value,
                "intent": case.intent.value,
                "question": case.question,
                "answer": resp.answer,
                "contexts": [chunk.text for chunk in chunks],
                "ground_truth": ", ".join(case.golden_answer_keywords),
                "actual_sources": resp.sources,
                "golden_sources": case.golden_sources,
                "forbidden_keywords": case.forbidden_keywords,
            }
        )

    metrics = run_ragas(rows) if rows else []
    for row, row_metrics in zip(rows, metrics, strict=True):
        row["ragas_metrics"] = row_metrics
        row["forbidden_check"] = forbidden_keywords_check(row["answer"], row["forbidden_keywords"])
        row["source_overlap"] = source_overlap(row["actual_sources"], row["golden_sources"])

    timestamp = datetime.datetime.now(datetime.UTC)
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"eval/results/{timestamp:%Y%m%dT%H%M%SZ}_{args.profile}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile": args.profile,
        "flags": flags.model_dump(),
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "filter": args.filter,
        "mode": args.mode,
        "rows": rows,
        "skipped": skipped,
        "aggregate": aggregate(rows),
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print_table(payload)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
