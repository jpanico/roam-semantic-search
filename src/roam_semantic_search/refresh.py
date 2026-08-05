"""Incremental index refresh: re-embed only what changed, delete what vanished.

A refresh fetches and normalizes the whole graph (cheap — around a second), then
diffs the resulting records' content hashes against the store: a record whose hash
is absent or different gets re-embedded and upserted, a stored uid no longer in the
graph is deleted, and everything else is untouched.  Hash diffing is the sole
selector — no timestamp heuristics — because a record's embeddable input can change
through *other* entities (an ancestor's text reaches descendants' breadcrumbs, a
referenced block's text reaches its referrers), which no per-entity timestamp can
see; the hash sees exactly what the embedding saw.

Public symbols:

- :class:`RefreshPlan` — the pure diff: records to (re-)embed, uids to delete.
- :class:`RefreshSummary` — what a refresh did, by count.
- :func:`refresh_plan` — diff normalized records against the store's hashes.
- :func:`refresh_store` — plan and apply a refresh against a store.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from guffin.roam.local_api import ApiEndpoint
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL, embed_texts
from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.normalize import IndexRecord, normalized_records
from roam_semantic_search.store import (
    SCHEMA_VERSION,
    StoreMeta,
    delete_records,
    read_meta,
    stamp_refresh,
    stored_hashes,
    upsert_records,
)

logger = logging.getLogger(__name__)


class RefreshPlan(BaseModel):
    """The pure diff between a graph's current records and a store's contents.

    Attributes:
        to_embed: Records that are new or whose embeddable input changed, in record order.
        deleted_uids: Stored uids no longer present among the current records.
        unchanged_count: Number of current records whose stored embedding is still valid.
    """

    model_config = ConfigDict(frozen=True)

    to_embed: tuple[IndexRecord, ...]
    deleted_uids: tuple[str, ...]
    unchanged_count: int


class RefreshSummary(BaseModel):
    """What a refresh did, by count.

    Attributes:
        embedded_count: Records re-embedded and upserted.
        deleted_count: Records deleted.
        unchanged_count: Records left untouched.
        record_count: Records in the store after the refresh.
    """

    model_config = ConfigDict(frozen=True)

    embedded_count: int
    deleted_count: int
    unchanged_count: int
    record_count: int


@validate_call
def refresh_plan(records: Sequence[IndexRecord], stored: Mapping[str, str]) -> RefreshPlan:
    """Diff the graph's current records against a store's content hashes.

    Args:
        records: The current normalized records of the whole graph.
        stored: The store's content hash by uid.

    Returns:
        The refresh plan: records to (re-)embed, uids to delete, unchanged count.
    """
    to_embed: Final[list[IndexRecord]] = [record for record in records if stored.get(record.uid) != record.content_hash]
    current_uids: Final[set[str]] = {record.uid for record in records}
    deleted_uids: Final[list[str]] = sorted(uid for uid in stored if uid not in current_uids)
    return RefreshPlan(
        to_embed=tuple(to_embed),
        deleted_uids=tuple(deleted_uids),
        unchanged_count=len(records) - len(to_embed),
    )


@validate_call
def refresh_store(
    db_path: Path,
    api_endpoint: ApiEndpoint,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    include_daily_notes: bool = True,
) -> RefreshSummary:
    """Plan and apply an incremental refresh against a store.

    Fetches and normalizes the whole graph, embeds exactly the records the plan
    selects (with the store's own model), upserts them, deletes vanished records,
    and stamps the refresh moment and record count into the store's meta.

    Args:
        db_path: The store's database file.
        api_endpoint: The Local API endpoint for the store's graph.
        ollama_url: The local embedding server's base URL.
        include_daily_notes: Whether daily-note pages (and their blocks) are indexed;
            pass what the store was built with, or records will be added/deleted
            accordingly.

    Returns:
        The refresh summary.

    Raises:
        ValueError: If the store's schema version is not the current one (a full
            rebuild is required — an incremental refresh cannot migrate the layout).
    """
    meta: Final[StoreMeta] = read_meta(db_path)
    if meta.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"store schema v{meta.schema_version} does not match current v{SCHEMA_VERSION}"
            " — rebuild the index with `roam-semantic-search build`"
        )
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    records: Final[list[IndexRecord]] = normalized_records(rows, include_daily_notes=include_daily_notes)
    plan: Final[RefreshPlan] = refresh_plan(records, stored_hashes(db_path))
    logger.info(
        "refresh plan: %d to embed, %d to delete, %d unchanged",
        len(plan.to_embed),
        len(plan.deleted_uids),
        plan.unchanged_count,
    )

    if plan.to_embed:
        vectors: Final[list[list[float]]] = embed_texts(
            [record.embed_input for record in plan.to_embed], model=meta.embed_model, base_url=ollama_url
        )
        embeddings: Final[NDArray[np.float32]] = np.asarray(vectors, dtype=np.float32)
        if int(embeddings.shape[1]) != meta.dimension:
            raise ValueError(
                f"model {meta.embed_model!r} now answers dimension {int(embeddings.shape[1])},"
                f" but the store holds {meta.dimension} — rebuild the index"
            )
        upsert_records(db_path, list(plan.to_embed), embeddings)
    if plan.deleted_uids:
        delete_records(db_path, list(plan.deleted_uids))

    record_count: Final[int] = len(records)
    stamp_refresh(db_path, record_count)
    return RefreshSummary(
        embedded_count=len(plan.to_embed),
        deleted_count=len(plan.deleted_uids),
        unchanged_count=plan.unchanged_count,
        record_count=record_count,
    )
