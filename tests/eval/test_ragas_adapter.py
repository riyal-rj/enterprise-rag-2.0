"""``ragas_adapter.run`` needs a live LLM to actually score rows, so it
isn't exercised end-to-end here. This only proves the module stays
importable even when the ``ragas`` package's own import chain is broken
(it currently is, in this environment — see ``ragas_adapter``'s module
docstring), since the ``ragas``/``datasets`` import is deferred to inside
``run`` rather than done at module load time.
"""

from app.eval import ragas_adapter


def test_module_imports_without_pulling_in_ragas() -> None:
    assert callable(ragas_adapter.run)
    assert ragas_adapter.METRIC_NAMES == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
