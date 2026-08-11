"""Wraps the ``ragas`` library to score each row's answer/context quality.

The ``ragas``/``datasets``/``langchain-community`` import chain is
deliberately deferred to inside :func:`run` rather than done at module load
time: it's a large, version-fragile dependency stack, and this module (and
everything that imports it, including the whole ``run_ragas`` CLI) should
stay importable even when that chain is broken. ``run`` is only ever called
with non-empty ``rows`` (see ``run_ragas.main``), so today — with no RAG
pipeline wired up yet (``ServiceInvoker`` always raises ``SkippedIntent``,
so ``rows`` stays empty) — this function is never actually invoked.
"""

from __future__ import annotations

from typing import Any, cast

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def run(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Compute RAGAS metrics for each row, in the same order as ``rows``.

    Each row must have ``question``, ``answer``, ``contexts`` (list[str]),
    and ``ground_truth`` keys (see ``run_ragas.main``'s row construction).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = Dataset.from_list(
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

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    scored = cast(Any, result).to_pandas()

    return [
        {name: float(scored.iloc[i][name]) for name in METRIC_NAMES if name in scored.columns}
        for i in range(len(rows))
    ]
