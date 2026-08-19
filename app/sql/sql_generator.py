"""SQL generation via Vanna, trained on the approved catalog/examples only.

Vanna (not this app's usual ``LLMClient.generate_structured`` pattern) is the
generator here per explicit product direction. It composes its own prompt
from whatever it has been *trained* on - DDL, documentation, and
question/SQL pairs stored in its own vector store - rather than taking a
rendered catalog string per call, so :meth:`VannaSQLGenerator.sync_catalog`/
:meth:`sync_examples` push the approved allowlist/examples into that store
out of band (catalog refresh, example approval), not on every request.

``allow_llm_to_see_data`` is never set True: letting the model run an
"intermediate" query against real data before the final query is generated
is exactly the kind of unbounded agency this feature's approval gate exists
to prevent (see the architecture blueprint's OWASP-risk mapping). Vanna's
own ``is_sql_valid`` is never consulted either - the AST policy
(``app.sql.sql_policy``) and a read-only Postgres ``EXPLAIN``
(``app.sql.sql_executor``) are the only trusted validators; Vanna is a
generation source, not a security boundary.
"""

from __future__ import annotations

import time
from typing import Protocol

from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from vanna.openai import OpenAI_Chat
from vanna.qdrant import Qdrant_VectorStore

from app.sql.catalog import StaticSQLCatalog
from app.sql.models import ApprovedSQLExample, CatalogSnapshot, SQLGeneration, SQLPrincipal


class SQLGenerator(Protocol):
    @property
    def cache_namespace(self) -> str: ...

    def generate(
        self,
        *,
        question: str,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
        rendered_catalog: str,
        examples: list[dict[str, str]],
        correction_code: str | None = None,
    ) -> SQLGeneration: ...


class _VannaClient(Qdrant_VectorStore, OpenAI_Chat):  # type: ignore[misc]
    """Vanna instance wired to this app's existing Qdrant/OpenAI clients.

    ``submit_prompt`` is overridden only to capture token usage -
    ``vanna.openai.OpenAI_Chat.submit_prompt`` calls the OpenAI SDK directly
    and discards ``response.usage`` entirely, but every other
    structured-generation call site in this codebase surfaces
    ``total_tokens`` (see ``app.core.llm.chat_client.StructuredLLMResponse``),
    and the RAG Ops panel's token-usage metrics depend on that being
    available consistently across features.

    ``generate_embedding`` is overridden for a different reason: Vanna
    0.7.9's own ``Qdrant_VectorStore.generate_embedding`` calls a private
    ``QdrantClient._get_or_init_model`` method that no longer exists on the
    ``qdrant-client>=1.17,<2`` this app already pins for its own hybrid
    search (see ``pyproject.toml`` and
    ``app.repositories.vector_repository``) - confirmed via
    ``AttributeError`` against the installed 1.19.0 client. Embedding
    directly through ``fastembed`` (already a dependency, and the same
    library Vanna's integration wraps internally) reuses the same model
    without depending on that private, version-fragile API.
    """

    def __init__(self, *, config: dict[str, object]) -> None:
        fastembed_model = config.get("fastembed_model", "BAAI/bge-small-en-v1.5")
        self._embedder = TextEmbedding(model_name=fastembed_model)  # type: ignore[arg-type]
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=config["_openai_client"], config=config)
        self.last_usage_tokens = 0

    def generate_embedding(self, data: str, **kwargs: object) -> list[float]:
        del kwargs
        embedding = next(iter(self._embedder.embed(data)))
        return embedding.tolist()  # type: ignore[no-any-return]

    def submit_prompt(self, prompt: list[dict[str, str]], **kwargs: object) -> str:
        model = kwargs.get("model") or self.config.get("model") if self.config else None
        response = self.client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=self.temperature,
            seed=self.config.get("seed") if self.config else None,
        )
        usage = response.usage
        self.last_usage_tokens = usage.total_tokens if usage else 0
        return response.choices[0].message.content or ""


class VannaSQLGenerator:
    def __init__(
        self,
        *,
        qdrant_client: QdrantClient,
        openai_client: OpenAI,
        model: str,
        temperature: float,
        seed: int,
        collection_prefix: str,
        fastembed_model: str,
        max_examples: int,
    ) -> None:
        self._vn = _VannaClient(
            config={
                "client": qdrant_client,
                "_openai_client": openai_client,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "fastembed_model": fastembed_model,
                "ddl_collection_name": f"{collection_prefix}_ddl",
                "documentation_collection_name": f"{collection_prefix}_documentation",
                "sql_collection_name": f"{collection_prefix}_sql",
                "n_results": max_examples,
            }
        )
        self._model = model
        self._prompt_version = "vanna-v1"
        self._synced_catalog_version: int | None = None

    @property
    def cache_namespace(self) -> str:
        return f"sql-generator:vanna:v1:model={self._model}"

    def sync_catalog(self, catalog: StaticSQLCatalog, snapshot: CatalogSnapshot) -> None:
        """Idempotently (re)train Vanna's DDL collection from the approved
        catalog. ``add_ddl`` upserts by a hash of the DDL text itself (see
        ``vanna.qdrant.Qdrant_VectorStore.add_ddl``), so calling this
        repeatedly with an unchanged catalog is a cheap no-op at the
        network/embedding level beyond the upsert round-trip. Only actually
        called when the version changed."""
        if self._synced_catalog_version == snapshot.version:
            return
        for statement in catalog.ddl_statements(snapshot):
            self._vn.train(ddl=statement)
        for rule in snapshot.business_rules:
            self._vn.train(documentation=rule)
        self._synced_catalog_version = snapshot.version

    def sync_examples(self, examples: list[ApprovedSQLExample]) -> None:
        """Idempotently (re)train Vanna's example collection - same upsert-
        by-content-hash guarantee as :meth:`sync_catalog` (see
        ``add_question_sql``)."""
        for example in examples:
            if example.active:
                self._vn.train(question=example.question, sql=example.sql)

    def generate(
        self,
        *,
        question: str,
        principal: SQLPrincipal,
        catalog: CatalogSnapshot,
        rendered_catalog: str,
        examples: list[dict[str, str]],
        correction_code: str | None = None,
    ) -> SQLGeneration:
        del principal, rendered_catalog, examples  # Vanna resolves its own context from training
        prompt_question = question
        if correction_code:
            prompt_question = (
                f"{question}\n\n"
                f"(A previous attempt at this question was rejected: {correction_code}. "
                "Generate a corrected, policy-compliant query.)"
            )
        started = time.perf_counter()
        sql = self._vn.generate_sql(prompt_question, allow_llm_to_see_data=False)
        duration_ms = (time.perf_counter() - started) * 1_000
        return SQLGeneration(
            sql=sql,
            # Vanna's generate_sql returns only the SQL string - table/
            # assumption self-reporting isn't part of its output shape.
            # app.sql.sql_policy.SQLPolicy independently derives the
            # authoritative referenced-table list from its own AST parse,
            # which is what ValidatedSQL/SQLProposal actually store.
            tables=(),
            assumptions=(),
            usage_tokens=self._vn.last_usage_tokens,
            duration_ms=duration_ms,
        )
