"""One-time migration: cut the app over to a Qdrant collection with named
dense + sparse (BM25) vectors, via a blue/green alias swap.

Default (safe) path::

    python -m scripts.migrate_qdrant_hybrid

Creates a new physical collection (``{QDRANT_COLLECTION}_v2``), ingests
every policy document into it via the same ``seed_docs()`` ingestion path
the app itself uses, requires a *complete* ``SeedReport`` (every
discovered file succeeded, and the collection's actual point count and
source set match what was written - not just "more than zero points"),
then repoints the app-facing collection name at it.

Qdrant forbids an alias sharing a name with an existing physical
collection. That matters for how the target name is freed up:

* If ``QDRANT_COLLECTION`` already names an alias (i.e. this script has
  been run before), the old alias entry is dropped and the new one
  created in a **single** ``update_collection_aliases()`` call - one
  atomic request, so there is no window where the name resolves to
  nothing.
* On the *very first* run, ``QDRANT_COLLECTION`` still names a bare,
  pre-migration physical collection. Reclaiming that literal name for an
  alias would require deleting the physical collection first - reopening
  exactly the not-found/race window ("a concurrent app instance
  auto-creates a bare replacement collection under that name") this
  migration exists to avoid. Instead, the first run creates a *new*,
  permanent alias name (``{QDRANT_COLLECTION}_live``) pointing at the
  freshly-seeded collection, and leaves the original bare collection
  untouched - no deletion, no gap, no race.

  This means the first run requires one manual follow-up: point
  ``QDRANT_COLLECTION`` at ``{QDRANT_COLLECTION}_live`` and redeploy the
  app. The script logs this instruction explicitly when it applies. The
  original bare collection is left in place (not deleted) so it can be
  removed manually once the app is confirmed running against the new
  alias.

Local-dev-only escape hatch (skips blue/green entirely, no rollback
copy)::

    python -m scripts.migrate_qdrant_hybrid --force-local-recreate

Deletes and recreates the collection in place. Never use this against a
collection anyone else might be querying.
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

from app.api.deps import (
    get_qdrant_client,
    get_query_cache_service,
    get_rag_ops_repository,
    get_semantic_query_cache,
)
from app.core.config import get_settings
from app.repositories.vector_repository import QdrantVectorRepository
from app.services.corpus_cache_invalidation_service import invalidate_corpus_caches
from scripts.seed_db import SeedReport, seed_docs

logger = logging.getLogger(__name__)

_LIVE_ALIAS_SUFFIX = "_live"


def force_local_recreate() -> None:
    """Delete and recreate the collection in place - local dev only, no rollback copy."""
    settings = get_settings().qdrant
    client = get_qdrant_client()
    name = settings.qdrant_collection
    if client.collection_exists(name):
        logger.warning("migrate.force_recreate_deleting | collection=%s", name)
        client.delete_collection(name)
    seed_docs()


def blue_green_migrate() -> None:
    settings = get_settings().qdrant
    client = get_qdrant_client()
    target_name = settings.qdrant_collection
    new_collection_name = f"{target_name}_v2"

    if client.collection_exists(new_collection_name):
        raise RuntimeError(
            f"'{new_collection_name}' already exists - delete it manually first if you "
            "want to re-run this migration from scratch (it's not auto-deleted, to avoid "
            "silently discarding a previous migration attempt's ingested data)."
        )

    logger.info("migrate.creating_collection | collection=%s", new_collection_name)
    new_repository = QdrantVectorRepository(client=client, collection_name=new_collection_name)

    logger.info("migrate.ingesting | collection=%s", new_collection_name)
    report = seed_docs(vector_repository=new_repository)
    _require_complete_corpus(report)

    point_count = client.count(new_collection_name).count
    actual_sources = {
        str(row["source"]) for row in new_repository.scroll_all_chunks() if row["source"]
    }
    logger.info(
        "migrate.ingestion_complete | collection=%s points=%d sources=%d",
        new_collection_name,
        point_count,
        len(actual_sources),
    )
    if point_count != report.chunks_written:
        raise RuntimeError(
            f"'{new_collection_name}' has {point_count} points but seed_docs() reported "
            f"writing {report.chunks_written} chunks - refusing to cut over on a mismatch "
            "between what was written and what's actually queryable."
        )
    if actual_sources != report.sources:
        raise RuntimeError(
            f"'{new_collection_name}' contains sources {sorted(actual_sources)}, but "
            f"seed_docs() reported ingesting {sorted(report.sources)} - refusing to cut "
            "over on a mismatch between the ingested corpus and what's actually stored."
        )

    _cutover_alias(client, target_name, new_collection_name)

    # The alias now resolves to freshly re-ingested content, but nothing
    # above touched corpus_version or the RAG-answer caches - without this,
    # a request landing right after cutover could still get an answer
    # cached (exact-match or semantic) from before the migration ever ran.
    # See app.services.corpus_cache_invalidation_service.
    invalidate_corpus_caches(
        rag_ops_repository=get_rag_ops_repository(),
        query_cache=get_query_cache_service(),
        semantic_query_cache=get_semantic_query_cache(),
        actor="qdrant_migration",
        source=f"{target_name}->{new_collection_name}",
    )


def _require_complete_corpus(report: SeedReport) -> None:
    if report.failed_files:
        raise RuntimeError(
            f"seed_docs() failed to ingest {len(report.failed_files)}/{report.discovered_files} "
            f"file(s): {report.failed_files} - refusing to cut over on a partial corpus. "
            "Check the seed.document_ingestion_failed logs above for details."
        )
    if report.succeeded_files != report.discovered_files:
        raise RuntimeError(
            f"seed_docs() only succeeded on {report.succeeded_files}/{report.discovered_files} "
            "discovered file(s) - refusing to cut over on a partial corpus."
        )
    if report.discovered_files == 0:
        raise RuntimeError(
            "seed_docs() discovered 0 files to ingest - refusing to cut over onto an empty "
            "collection. Check the policy directory path."
        )


def _cutover_alias(client: QdrantClient, target_name: str, new_collection_name: str) -> None:
    """Repoint ``target_name`` at ``new_collection_name``, atomically where
    possible (see the module docstring for why the very first run can't
    be fully atomic without introducing a new alias name)."""
    existing_aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    if target_name in existing_aliases:
        # Already alias-based from an earlier run of this script - delete
        # the old entry and create the new one in a single atomic call,
        # so there's no window where target_name resolves to nothing.
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=target_name)),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=new_collection_name, alias_name=target_name
                    )
                ),
            ]
        )
        logger.info(
            "migrate.cutover_complete | alias=%s -> %s (previous collection preserved for rollback)",
            target_name,
            new_collection_name,
        )
        return

    if not client.collection_exists(target_name):
        # Fresh install - nothing occupies target_name yet, so it can
        # become the alias directly.
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=new_collection_name, alias_name=target_name
                    )
                )
            ]
        )
        logger.info("migrate.cutover_complete | alias=%s -> %s", target_name, new_collection_name)
        return

    # First-ever migration: target_name is a bare physical collection, not
    # an alias. Qdrant forbids an alias sharing a name with an existing
    # collection, so claiming target_name itself would require deleting
    # it first - reopening the not-found/race window this migration
    # exists to avoid. Use a distinct, permanent alias name instead; the
    # original collection is left untouched (no deletion, no gap).
    live_alias = f"{target_name}{_LIVE_ALIAS_SUFFIX}"
    if live_alias in existing_aliases or client.collection_exists(live_alias):
        raise RuntimeError(
            f"'{live_alias}' already exists - resolve manually (rename/delete it) before "
            "re-running this migration."
        )
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=new_collection_name, alias_name=live_alias)
            )
        ]
    )
    logger.warning(
        "migrate.cutover_complete_action_required | alias=%s -> %s | "
        "original collection '%s' left untouched - set QDRANT_COLLECTION=%s and redeploy "
        "the app to actually start serving the new alias, then delete '%s' manually once "
        "traffic is confirmed on it",
        live_alias,
        new_collection_name,
        target_name,
        live_alias,
        target_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-local-recreate",
        action="store_true",
        help="Local dev only: delete and recreate in place, skipping blue/green (no rollback copy).",
    )
    args = parser.parse_args()

    if args.force_local_recreate:
        force_local_recreate()
    else:
        blue_green_migrate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
