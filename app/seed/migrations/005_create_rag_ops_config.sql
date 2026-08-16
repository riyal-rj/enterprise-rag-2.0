-- Centralized, mutable RAG operations config (feature-flag/config service),
-- replacing settings.rag's frozen (lru_cache'd, restart-only) defaults for
-- the fields an admin needs to toggle live from the RAG Operations panel
-- (see app.repositories.rag_ops_repository, app.controllers.rag_ops_controller,
-- app.api.deps.get_dynamic_reranker/get_rag_service/get_semantic_query_cache).
--
-- Singleton row (id always 1) - CHECK enforces at most one row ever exists.
-- Column defaults mirror app.core.config.rag_features.RAGFeatureSettings'
-- historical defaults, so a fresh deployment behaves exactly as before
-- until an admin actually touches the panel.

CREATE TABLE IF NOT EXISTS rag_ops_config (
    id                          SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    reranking_enabled           BOOLEAN NOT NULL DEFAULT FALSE,
    reranker_backend            VARCHAR(16) NOT NULL DEFAULT 'local'
                                    CHECK (reranker_backend IN ('local', 'voyage')),
    reranker_rollout_percentage SMALLINT NOT NULL DEFAULT 100
                                    CHECK (reranker_rollout_percentage BETWEEN 0 AND 100),
    semantic_cache_enabled      BOOLEAN NOT NULL DEFAULT FALSE,
    semantic_cache_threshold    REAL NOT NULL DEFAULT 0.95
                                    CHECK (semantic_cache_threshold BETWEEN 0 AND 1),
    emergency_disabled          BOOLEAN NOT NULL DEFAULT FALSE,
    emergency_disabled_reason   TEXT,
    emergency_disabled_at       TIMESTAMPTZ,
    emergency_disabled_by       VARCHAR(32),
    corpus_version              INTEGER NOT NULL DEFAULT 1,
    last_cache_invalidated_at   TIMESTAMPTZ,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by                  VARCHAR(32)
);

INSERT INTO rag_ops_config (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
