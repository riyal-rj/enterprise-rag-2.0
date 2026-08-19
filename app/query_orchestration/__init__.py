"""Query routing: decides whether a question goes to RAG, SQL, both, or
neither, ahead of either pipeline running. See
:class:`app.query_orchestration.query_orchestrator.QueryOrchestrator`.
"""

from __future__ import annotations
