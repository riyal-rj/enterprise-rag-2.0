"""Shared TypedDicts for the eval row/payload shapes.

These are the JSON-serializable dicts that flow between ``post_checks``,
``ragas_adapter``, ``run_ragas``, ``reporting``, and ``diff``. Typing them
here — instead of passing ``dict[str, Any]`` around — is what would have
caught the ``"hits"`` vs ``"found"`` key mismatch (a real bug hit earlier)
statically, via mypy/pyright, instead of only at runtime.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

MetricName = Literal["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class ForbiddenCheckResult(TypedDict):
    passed: bool
    found: list[str]


class SourceOverlapResult(TypedDict):
    passed: bool
    ratio: float
    matched: list[str]


class RetrievalMetrics(TypedDict):
    """Deterministic, rank-aware retrieval quality - computed from
    ``EvalRow.ranked_sources`` (retrieval order), not ``actual_sources``
    (which is alphabetically sorted - see ``ChatResponse.sources``)."""

    hit_rate: bool
    mrr: float


class RagasMetrics(TypedDict, total=False):
    """Per-row RAGAS scores. Every key is optional — a metric is absent
    when RAGAS couldn't or didn't score it for that row."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class EvalRow(TypedDict):
    """One golden case's invocation + scoring results.

    ``ragas_metrics``/``forbidden_check``/``source_overlap`` are absent
    until the scoring step runs (see ``run_ragas._score_rows``) — a freshly
    invoked row only has the first ten keys.
    """

    id: str
    demonstrates_feature: str
    intent: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    actual_sources: list[str]
    ranked_sources: list[str]
    golden_sources: list[str]
    forbidden_keywords: list[str]
    ragas_metrics: NotRequired[RagasMetrics]
    forbidden_check: NotRequired[ForbiddenCheckResult]
    source_overlap: NotRequired[SourceOverlapResult]
    retrieval_metrics: NotRequired[RetrievalMetrics]


class SkippedEntry(TypedDict):
    id: str
    reason: str


class AggregateResult(TypedDict):
    faithfulness: float | None
    context_precision: float | None
    context_recall: float | None
    answer_relevancy: float | None
    forbidden_violations: int
    hit_rate: float | None
    mrr: float | None


class EvalPayload(TypedDict):
    profile: str
    flags: dict[str, object]
    timestamp_utc: str
    filter: str | None
    mode: str
    rows: list[EvalRow]
    skipped: list[SkippedEntry]
    errors: list[SkippedEntry]
    aggregate: AggregateResult
