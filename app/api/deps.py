"""FastAPI dependency providers (composition root for DI).

Concrete implementations are constructed here and only here; every other
layer depends on the ``Protocol`` interfaces from ``app.core.security`` and
``app.repositories``, so swapping an implementation (e.g. Redis-backed rate
limiting) never touches service/controller code.
"""

from __future__ import annotations

from functools import lru_cache

from docling.chunking import HybridChunker
from fastapi import Depends
from openai import OpenAI
from upstash_redis import Redis

from app.controllers.admin_controller import AdminController
from app.controllers.auth_controller import AuthController
from app.core.config import Settings, get_settings
from app.core.db import PostgresConnectionPool
from app.core.ingestion.document_processor import (
    DoclingDocumentProcessor,
    DocumentProcessor,
    build_docling_converter,
)
from app.core.llm.chat_client import LLMClient, OpenAILLMClient, build_openai_client
from app.core.llm.embedding_client import EmbeddingClient, OpenAIEmbeddingClient
from app.core.redis_client import build_redis_client
from app.core.security.passwords import BcryptPasswordHasher, PasswordHasher
from app.core.security.rate_limiter import RateLimiter, UpstashSlidingWindowRateLimiter
from app.core.security.tokens import JWTTokenIssuer, JWTTokenVerifier, TokenIssuer, TokenVerifier
from app.repositories.user_repository import PostgresUserRepository, UserRepository
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
from app.services.query_cache_service import QueryCacheService, UpstashCacheBackend


@lru_cache(maxsize=1)
def get_db_pool() -> PostgresConnectionPool:
    """Process-wide pooled Postgres connection (lazily created, cached)."""
    return PostgresConnectionPool(get_settings().database.database_url)


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
def get_embedding_client() -> EmbeddingClient:
    return OpenAIEmbeddingClient(
        client=get_openai_client(),
        cache=get_query_cache_service(),
        default_model=get_settings().llm.embedding_model,
    )


@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    settings = get_settings().ingestion
    converter = build_docling_converter(
        settings.accelerator_device, settings.accelerator_num_threads
    )
    return DoclingDocumentProcessor(converter=converter, chunker=HybridChunker())


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
) -> AdminController:
    return AdminController(health_check_service, query_cache)
