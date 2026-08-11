"""Golden-case evaluation harness for the RAG/SQL pipeline.

Loads a curated set of golden Q&A cases (``app/eval/data/goldens.yaml``)
and grades a :class:`~app.eval.pipeline.QueryPipeline` implementation
against them. Run via ``python -m app.eval.run_ragas --profile <name>``
(see the Makefile's ``eval-*`` targets).
"""
