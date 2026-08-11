"""Golden-case evaluation harness for the RAG/SQL pipeline.

Loads a curated set of golden Q&A cases (``app/eval/data/goldens.yaml``),
invokes the system under test via an :class:`~app.eval.invokers.Invoker`
(service/in-process or, eventually, API), scores each answer with RAGAS
(:mod:`app.eval.ragas_adapter`) plus deterministic post-checks
(:mod:`app.eval.post_checks`), and aggregates/reports the results
(:mod:`app.eval.reporting`). Run via
``python -m app.eval.run_ragas --profile <name>`` (see the Makefile's
``eval-*`` targets); diff two runs with ``python -m app.eval.diff``.
"""
