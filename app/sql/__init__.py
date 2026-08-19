"""Text-to-SQL query route: a permission-aware, approval-gated tool path.

Distinct from :mod:`app.rag_services` (document evidence retrieval) - SQL
generation/execution has different authorization, freshness, correctness,
audit, and failure semantics, so it is a separate module behind
:class:`app.query_orchestration.query_orchestrator.QueryOrchestrator`, not a
code path inside ``RAGService``. See ``app.sql.sql_service.SQLService`` for
the orchestration entry point.
"""

from __future__ import annotations
