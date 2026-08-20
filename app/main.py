"""FastAPI application entrypoint (Factory pattern: ``create_app``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import (
    get_db_pool,
    get_llm_guard_input_scanner,
    get_llm_guard_output_scanner,
    get_sql_pool,
)
from app.api.rag_ops_sync import RagOpsConfigPoller
from app.api.routes.admin_routes import router as admin_router
from app.api.routes.auth_routes import router as auth_router
from app.api.routes.chat_routes import router as chat_router
from app.api.routes.rag_ops_routes import router as rag_ops_router
from app.api.routes.sql_routes import router as sql_router
from app.api.sql_proposal_recovery import StaleSQLProposalReclaimer
from app.core.config import Environment, get_settings
from app.core.exception_handlers import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # fail fast on invalid/missing config at startup

    # Guardrail ML scanners (PromptInjection, Anonymize, ...) are expensive
    # to construct (transformer model loads) - preload them once here so
    # the first real request never pays that cold-load latency, mirroring
    # the local-cross-encoder-reranker preload precedent. In production,
    # refuse to start rather than silently degrade to the deterministic-
    # only scanner layer if the operator hasn't confirmed models are
    # loadable in this deployment (SafetySettings.scanner_models_ready) -
    # see app.guardrails and app.core.config.safety.
    if settings.safety.scanner_models_ready:
        get_llm_guard_input_scanner()
        get_llm_guard_output_scanner()
    elif settings.environment is Environment.PRODUCTION:
        raise RuntimeError(
            "Refusing to start in production with SCANNER_MODELS_READY=false - "
            "the ML guardrail layer would silently run deterministic-only scanning. "
            "Set it once LLM Guard models are confirmed loadable in this deployment."
        )

    # Keeps this worker's RAG ops config in sync with config changes made
    # via another worker - see app.api.rag_ops_sync.
    poller = RagOpsConfigPoller()
    poller.start()
    # Recovers SQL proposals a crashed worker left stuck in EXECUTING - see
    # app.api.sql_proposal_recovery.
    sql_reclaimer = StaleSQLProposalReclaimer()
    sql_reclaimer.start()
    try:
        yield
    finally:
        await poller.stop()
        await sql_reclaimer.stop()
        if get_db_pool.cache_info().currsize:
            get_db_pool().close()
        if get_sql_pool.cache_info().currsize:
            sql_pool = get_sql_pool()
            if sql_pool is not None:
                sql_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Advanced RAG API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(rag_ops_router)
    app.include_router(chat_router)
    app.include_router(sql_router)
    return app


app = create_app()
