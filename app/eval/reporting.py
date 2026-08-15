"""Aggregates eval rows into summary stats and prints a markdown-table report."""

from __future__ import annotations

from statistics import mean
from typing import Literal, cast

from app.eval.types import AggregateResult, EvalPayload, EvalRow, MetricName


def aggregate(rows: list[EvalRow]) -> AggregateResult:
    """Mean of each RAGAS metric across ``rows``, plus a forbidden-keyword
    violation count.

    A metric is ``None`` when no row scored it (e.g. every case was
    skipped). ``forbidden_violations`` is a raw count, not a rate — for a
    security gate, "1 violation" is the signal that matters, not "99%
    passed".
    """
    return AggregateResult(
        faithfulness=_mean_metric(rows, "faithfulness"),
        context_precision=_mean_metric(rows, "context_precision"),
        context_recall=_mean_metric(rows, "context_recall"),
        answer_relevancy=_mean_metric(rows, "answer_relevancy"),
        forbidden_violations=sum(
            1 for row in rows if not row.get("forbidden_check", {}).get("passed", True)
        ),
        hit_rate=_mean_retrieval_metric(rows, "hit_rate"),
        mrr=_mean_retrieval_metric(rows, "mrr"),
    )


def _mean_metric(rows: list[EvalRow], metric: MetricName) -> float | None:
    values: list[float] = []
    for row in rows:
        metrics = row.get("ragas_metrics")
        if not metrics:
            continue
        value = metrics.get(metric)
        if value is not None:
            values.append(value)
    return round(mean(values), 3) if values else None


def _mean_retrieval_metric(rows: list[EvalRow], metric: Literal["hit_rate", "mrr"]) -> float | None:
    """Mean of a deterministic retrieval metric across rows.

    ``hit_rate`` is a bool per row - averaging it is exactly Hit Rate@K
    (the fraction of cases with at least one golden source retrieved).
    """
    values: list[float] = []
    for row in rows:
        retrieval = row.get("retrieval_metrics")
        if retrieval is None:
            continue
        values.append(float(retrieval[metric]))
    return round(mean(values), 3) if values else None


def _fmt(value: float | None) -> str:
    """Render a metric for display: ``N/A`` when unscored, never a bare ``0``."""
    return f"{value:.2f}" if value is not None else "N/A"


def print_table(payload: EvalPayload) -> None:
    """Print a markdown table of per-question results plus an aggregate row."""
    print(f"\n## Eval - profile={payload['profile']} mode={payload['mode']}")
    print(f"Skipped: {len(payload['skipped'])}")
    print()
    print("| id | feature | faith | ctx_prec | ctx_recall | ans_rel | forbidden |")
    print("|----|---------|-------|----------|------------|---------|-----------|")

    for row in payload["rows"]:
        metrics = row.get("ragas_metrics") or {}
        forbidden_check = cast(dict[str, object], row.get("forbidden_check") or {})
        forbidden = (
            "OK"
            if forbidden_check.get("passed", True)
            else f"FAIL: {forbidden_check.get('found', 'unknown')}"
        )
        print(
            f"| {row['id']} | {row['demonstrates_feature']} | "
            f"{_fmt(metrics.get('faithfulness'))} | "
            f"{_fmt(metrics.get('context_precision'))} | "
            f"{_fmt(metrics.get('context_recall'))} | "
            f"{_fmt(metrics.get('answer_relevancy'))} | {forbidden} |"
        )

    if not payload["rows"]:
        print("(no rows evaluated - all goldens skipped)")
        return

    agg = payload["aggregate"]
    print(
        f"| **AGG** | - | **{_fmt(agg['faithfulness'])}** | "
        f"**{_fmt(agg['context_precision'])}** | **{_fmt(agg['context_recall'])}** | "
        f"**{_fmt(agg['answer_relevancy'])}** | "
        f"violations={agg['forbidden_violations']} |"
    )
    print(f"hit_rate={_fmt(agg['hit_rate'])}  mrr={_fmt(agg['mrr'])}")
