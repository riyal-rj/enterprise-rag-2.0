"""One-time migration: cut the app over to a Qdrant collection with named
dense + sparse (BM25) vectors, via a blue/green alias swap.

Default (safe) path::

    python -m scripts.migrate_qdrant_hybrid

Creates a new physical collection (``{QDRANT_COLLECTION}_v2``), ingests
every policy document into it via the same ``seed_docs()`` ingestion path
the app itself uses, validates it has a non-zero point count, then
atomically repoints the app-facing ``QDRANT_COLLECTION``-named alias at it.

Qdrant has no native "rename collection" operation, so on the *first* run
against a pre-existing bare physical collection (not yet alias-based -
this app's actual starting state), freeing up that name for the alias
means preserving the old collection under a new name first. That's done
via Qdrant's own snapshot/recover mechanism (create a snapshot of the old
collection, recover it into ``{QDRANT_COLLECTION}_v1``, then delete the
original) - verified against a live Qdrant 1.17.0 instance during
planning, not assumed. The old data is never deleted without a verified
successful copy existing first.

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

from app.api.deps import get_qdrant_client
from app.core.config import get_settings
from app.repositories.vector_repository import QdrantVectorRepository
from scripts.seed_db import seed_docs

logger = logging.getLogger(__name__)


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
    seed_docs(vector_repository=new_repository)

    point_count = client.count(new_collection_name).count
    logger.info(
        "migrate.ingestion_complete | collection=%s points=%d", new_collection_name, point_count
    )
    if point_count == 0:
        raise RuntimeError(
            f"'{new_collection_name}' has 0 points after ingestion - refusing to cut over. "
            "Check the seed_docs() logs above for per-document failures."
        )

    _free_up_target_name(client, settings.qdrant_url, target_name)

    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=new_collection_name, alias_name=target_name)
            )
        ]
    )
    logger.info(
        "migrate.cutover_complete | alias=%s -> %s (previous collection preserved for rollback)",
        target_name,
        new_collection_name,
    )


def _free_up_target_name(client: QdrantClient, qdrant_url: str, target_name: str) -> None:
    """Make ``target_name`` available for the new alias, preserving whatever
    currently answers to it (a prior alias, or the original bare collection)."""
    existing_aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    if target_name in existing_aliases:
        # Already alias-based from an earlier run of this script - drop the
        # old alias entry only; the physical collection it pointed at is
        # untouched (that's the previous migration's rollback copy).
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=target_name))
            ]
        )
        return

    if not client.collection_exists(target_name):
        return  # nothing occupies the name yet - fresh install, nothing to preserve

    # First-ever migration: target_name is a bare physical collection, not
    # an alias. Qdrant can't rename it, so snapshot + recover it under a
    # new name before reclaiming target_name for the alias - never delete
    # the original until the copy is confirmed to exist.
    backup_name = f"{target_name}_v1"
    if client.collection_exists(backup_name):
        raise RuntimeError(
            f"Can't preserve '{target_name}' as '{backup_name}' - '{backup_name}' already "
            "exists. Resolve manually (rename/delete it) before re-running this migration."
        )

    logger.info("migrate.snapshotting_original | collection=%s", target_name)
    snapshot = client.create_snapshot(collection_name=target_name, wait=True)
    if snapshot is None:
        raise RuntimeError(f"Snapshot of '{target_name}' failed - refusing to delete it.")

    location = f"{qdrant_url}/collections/{target_name}/snapshots/{snapshot.name}"
    logger.info("migrate.restoring_as_backup | collection=%s", backup_name)
    recovered = client.recover_snapshot(collection_name=backup_name, location=location, wait=True)
    if not recovered or not client.collection_exists(backup_name):
        raise RuntimeError(
            f"Restoring '{target_name}' as '{backup_name}' failed - refusing to delete "
            f"the original. ('{target_name}' is untouched.)"
        )

    logger.info(
        "migrate.deleting_original | collection=%s (preserved as %s)", target_name, backup_name
    )
    client.delete_collection(target_name)


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
