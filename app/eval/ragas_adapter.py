"""Wraps the ``ragas`` library to score each row's answer/context quality.

The ``ragas``/``datasets``/``langchain-community`` import chain is
deliberately deferred to inside :func:`run` rather than done at module load
time: it's a large, version-fragile dependency stack, and this module (and
everything that imports it, including the whole ``run_ragas`` CLI) should
stay importable even when that chain is broken. ``run`` is only ever called
with non-empty ``rows`` (see ``run_ragas.main``), so today — with no RAG
pipeline wired up yet (``ServiceInvoker`` always raises ``SkippedIntent``,
so ``rows`` stays empty) — this function is never actually invoked.

The actual scoring call (``ragas.evaluate``) is a real, synchronous,
external LLM-judge call: it can hang or fail transiently, so it runs behind
a timeout + retry wrapper rather than a bare call.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from app.eval.types import EvalRow, MetricName, RagasMetrics

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

METRIC_NAMES: tuple[MetricName, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
_EVALUATE_TIMEOUT_SECONDS = 120.0


def run(rows: list[EvalRow]) -> list[RagasMetrics]:
    """Compute RAGAS metrics for each row, in the same order as ``rows``.

    Raises:
        Exception: whatever ``ragas.evaluate`` last raised, if every retry
            attempt fails or each attempt times out. Callers that can't
            afford to lose already-invoked rows over a scoring failure
            should catch this (see ``run_ragas._score_rows``).
    """
    from datasets import Dataset
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = _build_dataset(rows, Dataset)
    result = _evaluate_with_resilience(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    scored = cast(Any, result).to_pandas()

    return [
        cast(
            RagasMetrics,
            {name: float(scored.iloc[i][name]) for name in METRIC_NAMES if name in scored.columns},
        )
        for i in range(len(rows))
    ]


def _build_dataset(rows: list[EvalRow], dataset_cls: type[Dataset]) -> Dataset:
    return dataset_cls.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"] or [""],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )


def _evaluate_with_resilience(
    dataset: Dataset,
    metrics: list[Any],
    evaluate_fn: Callable[..., Any] | None = None,
) -> Any:
    """Run ``ragas.evaluate`` with a hard timeout, retrying transient failures.

    A thread-based timeout (not ``signal.alarm``) so this works on Windows
    too, where ``run_ragas`` is developed and run locally. Each attempt gets
    its own executor, shut down with ``wait=False``: Python can't forcibly
    kill a running thread, so a hung call keeps running in the background
    after its timeout fires — using ``wait=False`` (rather than the default
    ``with ThreadPoolExecutor()`` block, which calls ``shutdown(wait=True)``
    on exit) is what stops that background thread from silently negating
    the timeout by blocking here anyway.

    ``evaluate_fn`` defaults to ``ragas.evaluate``; overridable so this can
    be unit-tested without importing ``ragas`` itself.
    """
    resolved_evaluate_fn: Callable[..., Any]
    if evaluate_fn is not None:
        resolved_evaluate_fn = evaluate_fn
    else:
        from ragas import evaluate as resolved_evaluate_fn

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(resolved_evaluate_fn, dataset, metrics=metrics)
            return future.result(timeout=_EVALUATE_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - retry any transient scoring failure
            last_error = exc
            logger.warning(
                "eval.ragas_evaluate_retry",
                extra={"attempt": attempt, "max_attempts": _MAX_ATTEMPTS, "error": str(exc)},
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        finally:
            executor.shutdown(wait=False)

    assert last_error is not None  # loop always sets it before exhausting attempts
    raise last_error
