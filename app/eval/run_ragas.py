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
import logging
import sys
from pathlib import Path
from typing import Any, cast

from app.eval.invokers import Invoker, InvokeResponse, RetrievedChunk, ServiceInvoker, SkippedIntent
from app.eval.loader import load_golden_cases
from app.eval.post_checks import forbidden_keywords_check, retrieval_metrics, source_overlap
from app.eval.profiles import PROFILES, PipelineProfile
from app.eval.ragas_adapter import run as score_with_ragas
from app.eval.reporting import aggregate, print_table
from app.eval.schemas import GoldenCase
from app.eval.types import EvalPayload, EvalRow, SkippedEntry

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run golden eval cases against a pipeline profile."
    )
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
    parser.add_argument(
        "--mode", default="service", choices=["service", "api"], help="Invocation mode"
    )
    parser.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Fail when any selected case is skipped.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: eval/results/<timestamp>_<profile>.json)",
    )
    return parser.parse_args()


def _load_cases(questions_path: str | None, feature_filter: str | None) -> list[GoldenCase]:
    """Load goldens, optionally narrowed to one feature (+ baseline as a control set)."""
    cases = load_golden_cases(Path(questions_path) if questions_path else None)
    if feature_filter is None:
        return cases
    return [
        case for case in cases if case.demonstrates_feature.value in (feature_filter, "baseline")
    ]


def _select_invoker(mode: str) -> Invoker:
    """Build the transport into the system under test.

    Raises:
        NotImplementedError: for ``mode="api"`` — not built yet (Phase B).
    """
    if mode == "service":
        return ServiceInvoker()
    raise NotImplementedError(f"{mode!r} mode not yet implemented (Phase B)")


def _build_row(case: GoldenCase, response: InvokeResponse, chunks: list[RetrievedChunk]) -> EvalRow:
    return {
        "id": case.id,
        "demonstrates_feature": case.demonstrates_feature.value,
        "intent": case.intent.value,
        "question": case.question,
        "answer": response.answer,
        "contexts": [chunk.text for chunk in chunks],
        "ground_truth": ", ".join(case.golden_answer_keywords),
        "actual_sources": response.sources,
        # In actual retrieval-rank order, unlike `actual_sources` (an
        # alphabetically sorted set) - needed for rank-aware metrics
        # (MRR). See app.eval.post_checks.retrieval_metrics.
        "ranked_sources": [chunk.source for chunk in chunks],
        "golden_sources": case.golden_sources,
        "forbidden_keywords": case.forbidden_keywords,
    }


def _invoke_all(
    invoker: Invoker, cases: list[GoldenCase], flags: PipelineProfile
) -> tuple[list[EvalRow], list[SkippedEntry], list[SkippedEntry]]:
    """Call ``invoker`` for every case, sorting outcomes into three
    disjoint buckets - not two.

    ``skipped`` is for :class:`SkippedIntent`: a case this profile
    *intentionally* doesn't run (sql/hybrid need human approval,
    web_fallback has no pipeline, a profile requests an unimplemented
    feature). ``errors`` is for anything else - a genuine invocation
    failure. The two must stay apart: if they're merged, a regression
    that breaks every case in a profile is indistinguishable from a
    profile that legitimately has nothing applicable to run, and
    ``aggregate()`` (which only sees ``rows``) would silently report an
    empty-but-passing run instead of a failing one. See ``main()``, which
    fails the run (non-zero exit) when ``errors`` is non-empty."""
    rows: list[EvalRow] = []
    skipped: list[SkippedEntry] = []
    errors: list[SkippedEntry] = []

    for case in cases:
        try:
            response, chunks = invoker.invoke(case.question, flags, case.intent)
        except SkippedIntent as exc:
            logger.info("eval.case_skipped", extra={"case_id": case.id, "reason": str(exc)})
            skipped.append({"id": case.id, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - one bad case shouldn't kill the run
            logger.warning("eval.case_errored", extra={"case_id": case.id, "error": str(exc)})
            errors.append({"id": case.id, "reason": str(exc)})
            continue

        rows.append(_build_row(case, response, chunks))

    return rows, skipped, errors


def _score_rows(rows: list[EvalRow]) -> None:
    """Attach RAGAS metrics and post-checks to each row, in place.

    A ``ragas`` failure (a real external LLM-judge call, already retried
    internally — see ``ragas_adapter``) must not discard the rows already
    collected from successful invocations, so it's caught here and scored
    rows just end up with empty ``ragas_metrics`` instead of the whole run
    losing its output.
    """
    try:
        metrics = score_with_ragas(rows) if rows else []
    except Exception:
        logger.exception("eval.ragas_scoring_failed", extra={"row_count": len(rows)})
        metrics = [{} for _ in rows]

    for row, row_metrics in zip(rows, metrics, strict=True):
        # row_metrics may be an empty dict on scoring failure; cast to Any to
        # satisfy the TypedDict assignment without altering runtime behavior.
        row["ragas_metrics"] = cast(Any, row_metrics)
        row["forbidden_check"] = forbidden_keywords_check(row["answer"], row["forbidden_keywords"])
        row["source_overlap"] = source_overlap(row["actual_sources"], row["golden_sources"])
        row["retrieval_metrics"] = retrieval_metrics(row["ranked_sources"], row["golden_sources"])


def _resolve_output_path(output: str | None, profile: str, timestamp: datetime.datetime) -> Path:
    if output:
        return Path(output)
    return Path(f"eval/results/{timestamp:%Y%m%dT%H%M%SZ}_{profile}.json")


def _build_payload(
    args: argparse.Namespace,
    flags: PipelineProfile,
    timestamp: datetime.datetime,
    rows: list[EvalRow],
    skipped: list[SkippedEntry],
    errors: list[SkippedEntry],
) -> EvalPayload:
    return {
        "profile": args.profile,
        "flags": flags.model_dump(),
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "filter": args.filter,
        "mode": args.mode,
        "rows": rows,
        "skipped": skipped,
        "errors": errors,
        "aggregate": aggregate(rows),
    }


def main() -> None:
    args = _parse_args()
    flags = PROFILES[args.profile]
    cases = _load_cases(args.questions, args.filter)

    try:
        invoker = _select_invoker(args.mode)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    rows, skipped, errors = _invoke_all(invoker, cases, flags)
    _score_rows(rows)

    timestamp = datetime.datetime.now(datetime.UTC)
    out_path = _resolve_output_path(args.output, args.profile, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(args, flags, timestamp, rows, skipped, errors)
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info(
        "eval.run_complete",
        extra={
            "profile": args.profile,
            "rows": len(rows),
            "skipped": len(skipped),
            "errors": len(errors),
        },
    )
    print_table(payload)
    print(f"\nWrote: {out_path}")

    failed = False
    if errors:
        # Non-zero exit so CI actually fails on a broken profile, instead
        # of a run that errored on every case looking like "0 applicable
        # cases, all clean" (see _invoke_all's docstring).
        failed = True
        print(f"\n{len(errors)} case(s) errored (not skipped):", file=sys.stderr)
        for entry in errors:
            print(f"  {entry['id']}: {entry['reason']}", file=sys.stderr)
    if cases and not rows:
        failed = True
        print("\nNo selected case reached scoring.", file=sys.stderr)
    if args.fail_on_skip and skipped:
        failed = True
        print(f"\n{len(skipped)} selected case(s) were skipped:", file=sys.stderr)
        for entry in skipped:
            print(f"  {entry['id']}: {entry['reason']}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
