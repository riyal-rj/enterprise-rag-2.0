"""Text-to-SQL feature configuration.

Replaces the old ``VannaSettings`` (3 fields, unused - no separate DSN, no
vector-store config, nothing actually wired to a running feature). Every
non-negotiable control from the architecture blueprint that has a numeric
knob lives here: statement/lock timeouts, plan cost/row caps, result row/
byte/cell caps, proposal TTL, and the schema/function allowlists the AST
policy (``app.sql.sql_policy``) enforces.
"""

from __future__ import annotations

from pydantic import Field, SecretStr

from app.core.config.base import EnvBaseSettings

# Read-only, side-effect-free scalar functions. abs/extract/date_part/age/
# current_timestamp/current_date/greatest/least cover the date-math and
# magnitude comparisons generators commonly reach for ("opened in the last
# year", "largest transaction by amount") - current_timestamp/current_date
# specifically are what sqlglot's qualify() normalizes now()/current_date to
# for Postgres (see app.sql.sql_policy._build_qualify_schema's caller).
_DEFAULT_ALLOWED_FUNCTIONS = (
    "count,sum,avg,min,max,date_trunc,coalesce,nullif,round,lower,upper,"
    "abs,extract,date_part,age,current_timestamp,current_date,greatest,least,"
    "case,cast"
)


class SQLFeatureSettings(EnvBaseSettings):
    # Separate DSN/credential - never the app's own DatabaseSettings.database_url.
    # Empty by default: no analytics database has been provisioned for this
    # deployment yet (see scripts/sql/provision_sql_reader_role.sql). SQL
    # execution stays impossible without this set, regardless of any
    # RAG Ops admin toggle.
    sql_database_url: SecretStr = Field(default=SecretStr(""))
    sql_pool_min_conn: int = Field(default=1, ge=1)
    sql_pool_max_conn: int = Field(default=5, ge=1)

    sql_enabled_by_default: bool = Field(default=False)

    # Generation (Vanna) - see app.sql.sql_generator.
    sql_generator_model: str = Field(default="gpt-4o", min_length=1)
    sql_generator_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    sql_generator_seed: int = Field(default=42)
    sql_max_generation_attempts: int = Field(default=2, ge=1, le=3)
    sql_vanna_collection_prefix: str = Field(default="sql_vanna", min_length=1)
    sql_vanna_fastembed_model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    sql_max_examples: int = Field(default=10, ge=1, le=50)

    # Routing - see app.query_orchestration.intent_router.
    sql_router_model: str = Field(default="gpt-4o-mini", min_length=1)
    sql_router_prompt_version: str = Field(default="sql-router-v1", min_length=1)
    sql_min_route_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    sql_router_timeout_seconds: float = Field(default=8.0, gt=0.0, le=30.0)
    sql_router_max_completion_tokens: int = Field(default=200, ge=32, le=1_000)

    # AST policy - see app.sql.sql_policy.
    sql_policy_version: str = Field(default="sql-policy-v1", min_length=1)
    sql_max_joins: int = Field(default=5, ge=0, le=10)
    # Comma-separated, not JSON, for the same reason
    # ExternalAPISettings.crag_allowed_regulatory_domains is - pydantic-
    # settings parses list-typed env vars as JSON by default, an awkward
    # footgun for a plain SQL_ALLOWED_SCHEMAS=schema_a,schema_b value.
    sql_allowed_schemas_raw: str = Field(default="approved_analytics")
    sql_allowed_functions_raw: str = Field(default=_DEFAULT_ALLOWED_FUNCTIONS)

    # Execution - see app.sql.sql_executor.
    sql_statement_timeout_ms: int = Field(default=5_000, ge=100, le=30_000)
    sql_lock_timeout_ms: int = Field(default=500, ge=50, le=5_000)
    sql_max_plan_cost: float = Field(default=100_000.0, gt=0.0)
    sql_max_plan_rows: int = Field(default=100_000, ge=1)
    sql_max_result_rows: int = Field(default=200, ge=1, le=1_000)
    sql_max_result_bytes: int = Field(default=250_000, ge=1_000, le=2_000_000)
    sql_max_cell_chars: int = Field(default=2_000, ge=100, le=10_000)

    # Proposal/approval - see app.sql.sql_service.
    sql_proposal_ttl_seconds: int = Field(default=300, ge=30, le=900)

    @property
    def sql_allowed_schemas(self) -> frozenset[str]:
        return frozenset(
            schema.strip().casefold()
            for schema in self.sql_allowed_schemas_raw.split(",")
            if schema.strip()
        )

    @property
    def sql_allowed_functions(self) -> frozenset[str]:
        return frozenset(
            function.strip().casefold()
            for function in self.sql_allowed_functions_raw.split(",")
            if function.strip()
        )
