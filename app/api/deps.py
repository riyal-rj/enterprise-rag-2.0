"""FastAPI dependency providers (composition root for DI).

Concrete implementations are constructed here and only here; every other
layer depends on the ``Protocol`` interfaces from ``app.core.security`` and
``app.repositories``, so swapping an implementation (e.g. Redis-backed rate
limiting) never touches service/controller code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from docling.chunking import HybridChunker
from fastapi import Depends
from openai import OpenAI
from qdrant_client import QdrantClient
from upstash_redis import Redis

from app.controllers.admin_controller import AdminController
from app.controllers.auth_controller import AuthController
from app.controllers.chat_controller import ChatController
from app.controllers.rag_ops_controller import RagOpsController
from app.core.config import Settings, get_settings
from app.core.db import PostgresConnectionPool
from app.core.ingestion.document_processor import (
    DoclingDocumentProcessor,
    DocumentProcessor,
    build_docling_converter,
)
from app.core.llm.chat_client import LLMClient, OpenAILLMClient, build_openai_client
from app.core.llm.embedding_client import EmbeddingClient, OpenAIEmbeddingClient
from app.core.llm.sparse_embedding_client import FastEmbedSparseClient, SparseEmbeddingClient
from app.core.redis_client import build_redis_client
from app.core.security.passwords import BcryptPasswordHasher, PasswordHasher
from app.core.security.rate_limiter import RateLimiter, UpstashSlidingWindowRateLimiter
from app.core.security.tokens import JWTTokenIssuer, JWTTokenVerifier, TokenIssuer, TokenVerifier
from app.rag_services.cross_encoder_reranker import LocalCrossEncoderReranker
from app.rag_services.dynamic_query_transformer import DynamicQueryTransformer
from app.rag_services.dynamic_reranker import DynamicReranker
from app.rag_services.hyde_query_transformer import HydeQueryTransformer
from app.rag_services.query_transformer import FailOpenQueryTransformer, QueryTransformer
from app.rag_services.rag_runtime_config import RagRuntimeConfig, RagRuntimeConfigStore
from app.rag_services.rag_service import RAGService
from app.rag_services.reranker import FailOpenReranker, NoOpReranker, ReRanker
from app.rag_services.retrieval_strategy import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    RetrievalStrategy,
    SparseRetrievalStrategy,
)
from app.rag_services.voyage_reranker import VoyageReranker
from app.repositories.chat_history_repository import (
    ChatHistoryRepository,
    PostgresChatHistoryRepository,
)
from app.repositories.conversation_repository import (
    ConversationRepository,
    PostgresConversationRepository,
)
from app.repositories.rag_ops_repository import PostgresRagOpsRepository, RagOpsRepository
from app.repositories.semantic_cache_repository import QdrantSemanticQueryCache, SemanticQueryCache
from app.repositories.user_repository import PostgresUserRepository, UserRepository
from app.repositories.vector_repository import (
    QdrantVectorRepository,
    VectorRepository,
    build_qdrant_client,
)
from app.services.auth_service import AuthService
from app.services.health_checks import (
    HealthCheck,
    HealthCheckService,
    OpenAIHealthCheck,
    PostgresHealthCheck,
    QdrantHealthCheck,
    RedisHealthCheck,
    TavilyHealthCheck,
)
from app.services.policy_ingestion_service import PolicyIngestionService
from app.services.query_cache_service import QueryCacheService, UpstashCacheBackend
from app.services.rag_metrics_service import RagMetricsService

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_POLICY_DIR = _REPO_ROOT / "policy"


@lru_cache(maxsize=1)
def get_db_pool() -> PostgresConnectionPool:
    """Process-wide pooled Postgres connection (lazily created, cached)."""
    return PostgresConnectionPool(get_settings().database.database_url)


@lru_cache(maxsize=1)
def get_rag_ops_repository() -> RagOpsRepository:
    """Process-wide handle to the centralized, DB-backed RAG config/
    feature-flag store (see ``app.repositories.rag_ops_repository``) - the
    source of truth ``get_dynamic_reranker``/``get_rag_service``/
    ``get_semantic_query_cache`` seed their initial mutable state from at
    process start, instead of the frozen ``RAGFeatureSettings`` defaults."""
    return PostgresRagOpsRepository(get_db_pool())


@lru_cache(maxsize=1)
def get_rag_metrics_service() -> RagMetricsService:
    """Process-wide reranker/semantic-cache performance counters for the
    RAG Operations panel (lazily created, cached - see
    ``app.services.rag_metrics_service``)."""
    return RagMetricsService()


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """Process-wide Upstash Redis client, shared by every Redis-backed
    component (lazily created, cached)."""
    settings = get_settings()
    return build_redis_client(
        settings.cache.upstash_redis_url,
        settings.cache.upstash_redis_token.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Process-wide OpenAI SDK client, shared by chat completion and
    embeddings (lazily created, cached)."""
    return build_openai_client(get_settings().llm.openai_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return OpenAILLMClient(
        client=get_openai_client(),
        default_answer_model=settings.llm.llm_model_answer,
        default_grader_model=settings.llm.llm_model_grader,
    )


@lru_cache(maxsize=1)
def get_hyde_transformer() -> QueryTransformer:
    """Raw HyDE transformer: prompt/schema generation, batch embedding
    fusion, no rollout/emergency state. Never used directly against real
    ``/chat`` traffic - see ``get_dynamic_hyde_transformer`` (production)
    and ``get_eval_hyde_transformer`` (offline eval) below, which each wrap
    this for the availability/isolation guarantees their callers need."""
    settings = get_settings().rag
    return HydeQueryTransformer(
        llm_client=get_llm_client(),
        model=settings.hyde_model,
        prompt_version=settings.hyde_prompt_version,
        num_hypotheses=settings.hyde_num_hypotheses,
        temperature=settings.hyde_temperature,
        max_completion_tokens=settings.hyde_max_completion_tokens,
        timeout_seconds=settings.hyde_timeout_seconds,
        max_attempts=settings.hyde_max_attempts,
    )


@lru_cache(maxsize=1)
def get_eval_hyde_transformer() -> QueryTransformer:
    """HyDE transformer for the offline eval harness only.

    Deliberately not the same instance as ``get_dynamic_hyde_transformer``:
    eval profiles control HyDE enablement per case
    (``PipelineProfile.enable_hyde``, see ``app.eval.invokers``) and must
    not inherit an admin's live production rollout percentage or emergency
    state - an eval run has to be able to exercise HyDE for real regardless
    of what's currently rolled out to real traffic. Still fail-open, same
    availability guarantee as production.
    """
    return FailOpenQueryTransformer(get_hyde_transformer())


@lru_cache(maxsize=1)
def get_dynamic_hyde_transformer() -> DynamicQueryTransformer:
    """The HyDE transformer that actually serves real ``/chat`` traffic -
    admin-mutable (rollout%/emergency-disable) via the RAG Operations panel.
    Holds no config/metrics state of its own - ``RAGService.answer`` passes
    in the one ``RagRuntimeConfig`` snapshot it captures per request (see
    ``app.rag_services.rag_runtime_config``) and records HyDE metrics
    itself, since only it observes the complete generation-embedding-fusion
    pipeline."""
    return DynamicQueryTransformer(delegate=get_hyde_transformer())


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return OpenAIEmbeddingClient(
        client=get_openai_client(),
        cache=get_query_cache_service(),
        default_model=get_settings().llm.embedding_model,
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Process-wide Qdrant SDK client (lazily created, cached)."""
    settings = get_settings().qdrant
    return build_qdrant_client(settings.qdrant_url, settings.qdrant_timeout_seconds)


@lru_cache(maxsize=1)
def get_vector_repository() -> VectorRepository:
    settings = get_settings().qdrant
    return QdrantVectorRepository(
        client=get_qdrant_client(), collection_name=settings.qdrant_collection
    )


@lru_cache(maxsize=1)
def get_sparse_embedding_client() -> SparseEmbeddingClient:
    """Process-wide FastEmbed BM25 client (loads a local model - expensive
    to reconstruct per call, so cached like the other clients here)."""
    return FastEmbedSparseClient()


@lru_cache(maxsize=1)
def get_retrieval_strategies() -> dict[str, RetrievalStrategy]:
    """Every retrieval strategy the API can serve, keyed by ``RetrievalMode``.

    All three are always built - the ``hybrid_search_enabled`` flag doesn't
    decide which strategies exist, only which ones ``get_rag_service``'s
    ``allowed_retrieval_modes`` currently permits a real HTTP request to
    resolve to (see ``RAGService.answer``) - the eval harness needs every
    strategy constructible regardless of that flag.
    """
    settings = get_settings().rag
    vector_repository = get_vector_repository()
    sparse_embedding_client = get_sparse_embedding_client()

    dense: RetrievalStrategy = DenseRetrievalStrategy(vector_repository=vector_repository)
    sparse: RetrievalStrategy = SparseRetrievalStrategy(
        vector_repository=vector_repository, sparse_embedding_client=sparse_embedding_client
    )
    hybrid: RetrievalStrategy = HybridRetrievalStrategy(
        vector_repository=vector_repository,
        sparse_embedding_client=sparse_embedding_client,
        rrf_k=settings.rrf_k,
        candidate_top_k=settings.hybrid_candidate_top_k,
    )
    return {dense.name: dense, sparse.name: sparse, hybrid.name: hybrid}


@lru_cache(maxsize=1)
def get_reranker() -> ReRanker:
    """Process-wide reranker backend, fixed to ``RAGFeatureSettings.reranker_backend``
    (lazily created - the local cross-encoder loads model weights on first
    use, cached like the other model-backed clients here). Wrapped in
    ``FailOpenReranker`` so a reranker failure degrades to retrieval
    ordering instead of a 5xx.

    Deliberately static, unlike ``get_dynamic_reranker`` (which serves real
    ``/chat`` traffic): the eval harness (``app.eval.invokers.ServiceInvoker``)
    depends on this reranker actually running whenever a case's
    ``enable_rerank`` flag is on, not being silently bypassed by an admin's
    live rollout-percentage/emergency-disable setting - see
    ``app.rag_services.dynamic_reranker`` for why that bypass exists at all.
    """
    settings = get_settings().rag
    delegate: ReRanker
    if settings.reranker_backend == "voyage":
        delegate = VoyageReranker(
            api_key=settings.voyage_api_key.get_secret_value(),
            model_name=settings.voyage_model,
        )
    else:
        delegate = LocalCrossEncoderReranker(model_name=settings.reranker_model)
    return FailOpenReranker(delegate=delegate, fallback=NoOpReranker())


@lru_cache(maxsize=1)
def get_rag_runtime_config_store() -> RagRuntimeConfigStore:
    """Process-wide, atomically-swapped snapshot of the admin-mutable RAG
    Ops fields (see ``app.rag_services.rag_runtime_config``) - the single
    shared source ``RAGService.answer`` reads from (once per request,
    passing it into ``DynamicReranker``/``DynamicQueryTransformer`` as a
    plain argument rather than letting either read the store itself), so an
    admin's config update (or ``RagOpsConfigPoller``'s cross-worker sync -
    see ``app.api.rag_ops_sync``) is applied to every collaborator in one
    atomic swap instead of via each object's own independently-timed
    mutations.

    ``hyde_enabled`` is deliberately stored raw here (not pre-baked with
    ``emergency_disabled``, unlike ``reranking_enabled``/
    ``semantic_cache_enabled``) - see
    ``app.controllers.rag_ops_controller.apply_rag_ops_config`` for why."""
    config = get_rag_ops_repository().get_config()
    return RagRuntimeConfigStore(
        RagRuntimeConfig(
            reranking_enabled=config.reranking_enabled and not config.emergency_disabled,
            reranker_backend=config.reranker_backend,  # type: ignore[arg-type]
            reranker_rollout_percentage=config.reranker_rollout_percentage,
            emergency_disabled=config.emergency_disabled,
            semantic_cache_enabled=config.semantic_cache_enabled and not config.emergency_disabled,
            semantic_cache_threshold=config.semantic_cache_threshold,
            corpus_version=config.corpus_version,
            hyde_enabled=config.hyde_enabled,
            hyde_rollout_percentage=config.hyde_rollout_percentage,
        )
    )


@lru_cache(maxsize=1)
def get_dynamic_reranker() -> DynamicReranker:
    """The reranker that actually serves real ``/chat`` traffic - admin-
    mutable (backend/rollout%/emergency-disable) via the RAG Operations
    panel, unlike ``get_reranker`` above. Both local and Voyage delegates
    are constructed up front (Voyage only if an API key is configured) so
    an admin can flip ``reranker_backend`` at any time without a cold
    construction cost on the next request. Holds no config state of its
    own - ``RAGService.answer`` passes in the one ``RagRuntimeConfig``
    snapshot it captures per request."""
    settings = get_settings().rag
    local: ReRanker = LocalCrossEncoderReranker(model_name=settings.reranker_model)
    voyage: ReRanker | None = None
    if settings.voyage_api_key.get_secret_value():
        voyage = VoyageReranker(
            api_key=settings.voyage_api_key.get_secret_value(),
            model_name=settings.voyage_model,
        )
    return DynamicReranker(local=local, voyage=voyage, metrics=get_rag_metrics_service())


@lru_cache(maxsize=1)
def get_semantic_query_cache() -> SemanticQueryCache:
    """Process-wide paraphrase-aware query cache (lazy-singleton shape as
    ``get_vector_repository``, but its Qdrant collection is only ensured to
    exist on first real use, not at construction time - see
    ``QdrantSemanticQueryCache``'s docstring - since this is always
    constructed regardless of whether semantic caching is currently
    enabled). Holds no config state of its own - ``RAGService.answer``
    passes ``similarity_threshold`` in per-call, from the same snapshot it
    uses for everything else in that request (see
    ``app.repositories.semantic_cache_repository``)."""
    settings = get_settings().rag
    return QdrantSemanticQueryCache(
        client=get_qdrant_client(),
        collection_name=settings.semantic_cache_collection,
    )


def get_default_retrieval_mode() -> str:
    return "hybrid" if get_settings().rag.hybrid_search_enabled else "dense"


def get_allowed_retrieval_modes() -> frozenset[str]:
    """Modes a real HTTP ``/chat`` request is currently allowed to resolve
    to - the rollout gate. ``dense`` is always allowed (the safe fallback);
    ``sparse``/``hybrid`` require ``hybrid_search_enabled``. The eval
    harness bypasses this entirely (see ``app.eval.invokers``), since it
    must be able to run every mode for real ahead of flipping this flag.
    """
    if get_settings().rag.hybrid_search_enabled:
        return frozenset(get_retrieval_strategies())
    return frozenset({"dense"})


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """The RAGService that serves real ``/chat`` traffic.

    ``reranking_enabled``/``semantic_cache_enabled``/``hyde_enabled``/
    ``corpus_version`` are all read live from the shared
    ``RagRuntimeConfigStore`` (not ``RAGFeatureSettings`` directly) - and
    the semantic cache instance is always passed in (regardless of whether
    it starts enabled), since an admin can flip ``semantic_cache_enabled``
    on later via that same store, which ``RAGService.answer`` additionally
    gates on a real cache instance having been supplied here at
    construction time.
    """
    settings = get_settings().rag
    return RAGService(
        embedding_client=get_embedding_client(),
        retrieval_strategies=get_retrieval_strategies(),
        llm_client=get_llm_client(),
        cache=get_query_cache_service(),
        default_retrieval_mode=get_default_retrieval_mode(),
        allowed_retrieval_modes=get_allowed_retrieval_modes(),
        reranker=get_dynamic_reranker(),
        reranker_initial_top_k=settings.reranker_initial_top_k,
        semantic_cache=get_semantic_query_cache(),
        metrics=get_rag_metrics_service(),
        query_transformer=get_dynamic_hyde_transformer(),
        config_store=get_rag_runtime_config_store(),
    )


@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    settings = get_settings().ingestion
    converter = build_docling_converter(
        settings.accelerator_device, settings.accelerator_num_threads
    )
    return DoclingDocumentProcessor(converter=converter, chunker=HybridChunker())


@lru_cache(maxsize=1)
def get_policy_ingestion_service() -> PolicyIngestionService:
    return PolicyIngestionService(
        document_processor=get_document_processor(),
        embedding_client=get_embedding_client(),
        sparse_embedding_client=get_sparse_embedding_client(),
        vector_repository=get_vector_repository(),
        policy_dir=_POLICY_DIR,
        max_upload_size_mb=get_settings().ingestion.max_upload_size_mb,
    )


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """Process-wide rate limiter singleton (Upstash-backed, shared across
    instances)."""
    return UpstashSlidingWindowRateLimiter(get_redis_client())


@lru_cache(maxsize=1)
def get_password_hasher() -> PasswordHasher:
    return BcryptPasswordHasher()


def get_token_issuer(settings: Settings = Depends(get_settings)) -> TokenIssuer:
    return JWTTokenIssuer(
        secret=settings.auth.jwt_secret.get_secret_value(),
        algorithm=settings.auth.jwt_algorithm,
        expires_minutes=settings.auth.jwt_expiration_minutes,
    )


def get_token_verifier(settings: Settings = Depends(get_settings)) -> TokenVerifier:
    return JWTTokenVerifier(
        secret=settings.auth.jwt_secret.get_secret_value(),
        algorithm=settings.auth.jwt_algorithm,
    )


def get_user_repository(
    pool: PostgresConnectionPool = Depends(get_db_pool),
) -> UserRepository:
    return PostgresUserRepository(pool)


def get_auth_service(
    settings: Settings = Depends(get_settings),
    user_repository: UserRepository = Depends(get_user_repository),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
    token_issuer: TokenIssuer = Depends(get_token_issuer),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
) -> AuthService:
    return AuthService(
        user_repository=user_repository,
        password_hasher=password_hasher,
        token_issuer=token_issuer,
        rate_limiter=rate_limiter,
        rate_limits=settings.rate_limit,
    )


def get_auth_controller(
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthController:
    return AuthController(auth_service)


def get_chat_history_repository(
    pool: PostgresConnectionPool = Depends(get_db_pool),
) -> ChatHistoryRepository:
    return PostgresChatHistoryRepository(pool)


def get_conversation_repository(
    pool: PostgresConnectionPool = Depends(get_db_pool),
) -> ConversationRepository:
    return PostgresConversationRepository(pool)


def get_chat_controller(
    rag_service: RAGService = Depends(get_rag_service),
    chat_history_repository: ChatHistoryRepository = Depends(get_chat_history_repository),
    conversation_repository: ConversationRepository = Depends(get_conversation_repository),
) -> ChatController:
    return ChatController(rag_service, chat_history_repository, conversation_repository)


def get_health_checks(
    settings: Settings = Depends(get_settings),
    pool: PostgresConnectionPool = Depends(get_db_pool),
    redis_client: Redis = Depends(get_redis_client),
) -> list[HealthCheck]:
    return [
        PostgresHealthCheck(pool),
        QdrantHealthCheck(settings.qdrant.qdrant_url),
        RedisHealthCheck(redis_client),
        OpenAIHealthCheck(settings.llm.openai_api_key.get_secret_value()),
        TavilyHealthCheck(),
    ]


def get_health_check_service(
    checks: list[HealthCheck] = Depends(get_health_checks),
) -> HealthCheckService:
    return HealthCheckService(checks)


@lru_cache(maxsize=1)
def get_query_cache_service() -> QueryCacheService:
    """Process-wide query cache singleton (lazily created, cached)."""
    backend = UpstashCacheBackend(get_redis_client())
    return QueryCacheService(backend, get_settings().cache)


def get_admin_controller(
    health_check_service: HealthCheckService = Depends(get_health_check_service),
    query_cache: QueryCacheService = Depends(get_query_cache_service),
    policy_ingestion_service: PolicyIngestionService = Depends(get_policy_ingestion_service),
    semantic_query_cache: SemanticQueryCache = Depends(get_semantic_query_cache),
    rag_ops_repository: RagOpsRepository = Depends(get_rag_ops_repository),
) -> AdminController:
    return AdminController(
        health_check_service,
        query_cache,
        policy_ingestion_service,
        semantic_query_cache,
        rag_ops_repository,
    )


def get_rag_ops_controller(
    repository: RagOpsRepository = Depends(get_rag_ops_repository),
    config_store: RagRuntimeConfigStore = Depends(get_rag_runtime_config_store),
    reranker: DynamicReranker = Depends(get_dynamic_reranker),
    metrics: RagMetricsService = Depends(get_rag_metrics_service),
) -> RagOpsController:
    return RagOpsController(repository, config_store, reranker, metrics)
