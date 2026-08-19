"""FastAPI application entrypoint (Factory pattern: ``create_app``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_db_pool, get_sql_pool
from app.api.rag_ops_sync import RagOpsConfigPoller
from app.api.routes.admin_routes import router as admin_router
from app.api.routes.auth_routes import router as auth_router
from app.api.routes.chat_routes import router as chat_router
from app.api.routes.rag_ops_routes import router as rag_ops_router
from app.api.routes.sql_routes import router as sql_router
from app.core.config import get_settings
from app.core.exception_handlers import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings()  # fail fast on invalid/missing config at startup
    # Keeps this worker's RAG ops config in sync with config changes made
    # via another worker - see app.api.rag_ops_sync.
    poller = RagOpsConfigPoller()
    poller.start()
    try:
        yield
    finally:
        await poller.stop()
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
