"""One-time local dev setup: apply SQL migrations, then ingest ``policy/`` into Qdrant.

Run via ``make seed`` (both steps) or ``make seed-docs`` (ingestion only,
once the DB is already migrated). Ingestion reuses the same DI providers
the running app uses (``app.api.deps.get_document_processor`` /
``get_embedding_client`` / ``get_vector_repository``) rather than building
its own clients, so a seeded document is chunked/embedded/stored exactly
as the app would do it at runtime - no separate code path to drift out of
sync with ``DoclingDocumentProcessor``/``QdrantVectorRepository``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg2

from app.api.deps import (
    get_document_processor,
    get_embedding_client,
    get_sparse_embedding_client,
    get_vector_repository,
)
from app.core.config import get_settings
from app.core.ingestion.document_processor import SUPPORTED_SUFFIXES
from app.repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _REPO_ROOT / "app" / "seed" / "migrations"
_POLICY_DIR = _REPO_ROOT / "policy"


def run_migrations() -> None:
    """Apply every ``*.sql`` file in ``app/seed/migrations``, in filename order.

    Each migration is written to be safe to re-run (``CREATE TABLE IF NOT
    EXISTS``, ``ON CONFLICT DO NOTHING``), so this doesn't track which
    migrations already ran - it just re-applies all of them.
    """
    settings = get_settings()
    migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        logger.warning("seed.no_migrations_found | dir=%s", _MIGRATIONS_DIR)
        return

    conn = psycopg2.connect(settings.database.database_url)
    try:
        with conn.cursor() as cur:
            for path in migration_files:
                logger.info("seed.applying_migration | file=%s", path.name)
                cur.execute(path.read_text())
        conn.commit()
    finally:
        conn.close()


def seed_docs(policy_dir: Path = _POLICY_DIR, vector_repository: VectorRepository | None = None) -> None:
    """Chunk, embed, and upsert every PDF/DOCX in ``policy_dir`` into Qdrant.

    ``vector_repository`` defaults to the DI-provided singleton (the
    app's configured collection); pass an explicit one to target a
    different collection - e.g. ``scripts/migrate_qdrant_hybrid.py``
    ingesting into a new physical collection ahead of an alias cutover,
    without a permanent env change.

    One document's processing failure (a corrupt file, an unsupported
    layout) is logged and skipped rather than aborting the whole run -
    each file is an independent unit of work in a 65-document batch job.
    """
    processor = get_document_processor()
    embedder = get_embedding_client()
    sparse_embedder = get_sparse_embedding_client()
    repository = vector_repository if vector_repository is not None else get_vector_repository()

    files = sorted(p for p in policy_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        logger.warning("seed.no_documents_found | dir=%s", policy_dir)
        return

    total_chunks = 0
    for path in files:
        try:
            chunks = processor.process_document(str(path))
            if not chunks:
                logger.warning("seed.no_chunks_extracted | file=%s", path.name)
                continue

            texts = [chunk.text for chunk in chunks]
            dense_embeddings = embedder.embed_texts(texts)
            sparse_embeddings = sparse_embedder.embed_documents(texts)
            repository.upsert_chunks(chunks, dense_embeddings, sparse_embeddings)
        except Exception:
            logger.exception("seed.document_ingestion_failed | file=%s", path.name)
            continue

        total_chunks += len(chunks)
        logger.info("seed.document_ingested | file=%s chunks=%d", path.name, len(chunks))

    logger.info(
        "seed.ingestion_complete | documents=%d chunks=%d", len(files), total_chunks
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_migrations()
    seed_docs()
