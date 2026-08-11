"""CLI: run golden eval cases against a pipeline profile.

    python -m app.eval.run_ragas --profile naive
    python -m app.eval.run_ragas --profile hybrid+rerank+hyde --filter hyde

See the Makefile's ``eval-*`` targets for the exact profile/filter
combinations used in CI/local workflows.
"""

from __future__ import annotations

import argparse
import asyncio

from app.eval.grading import KeywordSourceGrader
from app.eval.loader import filter_by_feature, load_golden_cases
from app.eval.pipeline import build_pipeline
from app.eval.profiles import PROFILES, get_profile
from app.eval.reporting import print_summary, write_report
from app.eval.runner import EvalRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden eval cases against a pipeline profile.")
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument(
        "--filter", default=None, help="Only run cases whose demonstrates_feature matches this value."
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    profile = get_profile(args.profile)
    cases = filter_by_feature(load_golden_cases(), args.filter)

    pipeline = build_pipeline(profile)
    runner = EvalRunner(pipeline, KeywordSourceGrader(), is_baseline=(profile.name == "naive"))
    report = await runner.run(profile.name, cases)

    print_summary(report)
    path = write_report(report)
    print(f"\nWrote {path}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
