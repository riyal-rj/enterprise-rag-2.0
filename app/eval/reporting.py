"""Aggregates eval rows into summary stats and prints a markdown-table report."""

from __future__ import annotations

from statistics import mean
from typing import Any

from app.eval.ragas_adapter import METRIC_NAMES as RAGAS_METRIC_NAMES


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each RAGAS metric across ``rows``, plus a forbidden-keyword
    violation count.

    A metric is ``None`` when no row scored it (e.g. every case was
    skipped). ``forbidden_violations`` is a raw count, not a rate — for a
    security gate, "1 violation" is the signal that matters, not "99%
    passed".
    """
    out: dict[str, Any] = {}
    for metric in RAGAS_METRIC_NAMES:
        values = [
            row["ragas_metrics"].get(metric)
            for row in rows
            if row.get("ragas_metrics") and row["ragas_metrics"].get(metric) is not None
        ]
        out[metric] = round(mean(values), 3) if values else None

    out["forbidden_violations"] = sum(
        1 for row in rows if not row.get("forbidden_check", {}).get("passed", True)
    )
    return out


def print_table(payload: dict[str, Any]) -> None:
    """Print a markdown table of per-question results plus an aggregate row."""
    print(f"\n## Eval - profile={payload['profile']} mode={payload['mode']}")
    print(f"Skipped: {len(payload['skipped'])}")
    print()
    print("| id | feature | faith | ctx_prec | ctx_recall | ans_rel | forbidden |")
    print("|----|---------|-------|----------|------------|---------|-----------|")

    for row in payload["rows"]:
        metrics = row.get("ragas_metrics") or {}
        forbidden = (
            "OK"
            if row["forbidden_check"]["passed"]
            else f"FAIL: {row['forbidden_check']['found']}"
        )
        print(
            f"| {row['id']} | {row['demonstrates_feature']} | "
            f"{metrics.get('faithfulness', 0):.2f} | "
            f"{metrics.get('context_precision', 0):.2f} | "
            f"{metrics.get('context_recall', 0):.2f} | "
            f"{metrics.get('answer_relevancy', 0):.2f} | {forbidden} |"
        )

    if not payload["rows"]:
        print("(no rows evaluated - all goldens skipped)")
        return

    agg = payload["aggregate"]
    print(
        f"| **AGG** | — | **{agg['faithfulness']}** | "
        f"**{agg['context_precision']}** | **{agg['context_recall']}** | "
        f"**{agg['answer_relevancy']}** | "
        f"violations={agg['forbidden_violations']} |"
    )
