"""Incremental index refresh: re-embed only what changed, delete what vanished.

A refresh fetches and normalizes the whole graph (cheap — around a second), then
diffs the resulting records' content hashes against the store: a record whose hash
is absent or different gets re-embedded and upserted, a stored uid no longer in the
graph is deleted, and everything else is untouched.  Records are identified by
record id, so each chunk of an attached PDF diffs on its own.

Attached PDFs are budgeted twice over, so an interactive refresh stays quick however large
a backlog is.  Given a :class:`~roam_semantic_search.pdf_text.PdfFetchBudget`, a refresh
first fetches the PDFs its PDF text cache lacks, up to that budget; and given
``max_pdf_embeds``, it embeds at most that many PDF chunk records, deferring the rest to
the next refresh.  Page and block records are never deferred: an edit always lands.  A
failure while fetching PDFs never fails the refresh — it is reported, and the refresh
proceeds on what the cache already holds.  Hash diffing is the sole
selector — no timestamp heuristics — because a record's embeddable input can change
through *other* entities (an ancestor's text reaches descendants' breadcrumbs, a
referenced block's text reaches its referrers), which no per-entity timestamp can
see; the hash sees exactly what the embedding saw.

Public symbols:

- :class:`RefreshPlan` — the pure diff: records to (re-)embed, record ids to delete.
- :class:`RefreshSummary` — what a refresh did, by count.
- :data:`INTERACTIVE_PDF_FETCH_BUDGET` / :data:`INTERACTIVE_MAX_PDF_EMBEDS` — the PDF budget
  of a refresh someone is waiting on (the MCP server's).
- :func:`budgeted_embeds` — split a plan's records into those embedded now and those deferred.
- :func:`refresh_plan` — diff normalized records against the store's hashes.
- :func:`refresh_store` — plan and apply a refresh against a store.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import requests
from guffin.roam.local_api import ApiEndpoint
from guffin.roam.local_api_source import LocalApiGraphSource
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL, embed_texts
from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.normalize import (
    IndexRecord,
    RecordKind,
    indexable_pdf_embeds,
    normalized_records,
    pdf_chunk_records,
)
from roam_semantic_search.pdf_text import (
    ExtractionSummary,
    PdfFetchBudget,
    cached_pdf_texts,
    default_pdf_cache_path,
    extract_pdf_texts,
)
from roam_semantic_search.store import (
    SCHEMA_VERSION,
    StoreMeta,
    delete_records,
    read_meta,
    stamp_refresh,
    stored_hashes,
    stored_record_count,
    upgrade_store,
    upsert_records,
)

logger = logging.getLogger(__name__)

INTERACTIVE_PDF_FETCH_BUDGET: Final[PdfFetchBudget] = PdfFetchBudget(
    max_pdfs=10, max_bytes=50_000_000, max_seconds=15.0
)
"""PDF fetching a refresh someone is waiting on may do: ~1 s per typical PDF, so ten at most."""

INTERACTIVE_MAX_PDF_EMBEDS: Final[int] = 250
"""PDF chunk embeddings a refresh someone is waiting on may do (~15 s at ~18 chunks/s)."""


class RefreshPlan(BaseModel):
    """The pure diff between a graph's current records and a store's contents.

    Attributes:
        to_embed: Records that are new or whose embeddable input changed, in record order.
        deleted_ids: Stored record ids no longer present among the current records.
        unchanged_count: Number of current records whose stored embedding is still valid.
    """

    model_config = ConfigDict(frozen=True)

    to_embed: tuple[IndexRecord, ...]
    deleted_ids: tuple[str, ...]
    unchanged_count: int


class RefreshSummary(BaseModel):
    """What a refresh did, by count.

    Attributes:
        embedded_count: Records re-embedded and upserted.
        deleted_count: Records deleted.
        unchanged_count: Records left untouched.
        record_count: Records in the store after the refresh.
        pdf_fetched_count: PDFs fetched (and their text cached) by this refresh.
        pdf_pending_count: PDFs the cache still lacks because the fetch budget ran out.
        embed_deferred_count: PDF chunk records left for a later refresh by the embed cap.
        pdf_error: Why fetching PDFs stopped early, when it failed; the refresh went on
            with what the cache already held.
    """

    model_config = ConfigDict(frozen=True)

    embedded_count: int
    deleted_count: int
    unchanged_count: int
    record_count: int
    pdf_fetched_count: int = 0
    pdf_pending_count: int = 0
    embed_deferred_count: int = 0
    pdf_error: str | None = None


@validate_call
def refresh_plan(records: Sequence[IndexRecord], stored: Mapping[str, str]) -> RefreshPlan:
    """Diff the graph's current records against a store's content hashes.

    Args:
        records: The current normalized records of the whole graph.
        stored: The store's content hash by record id.

    Returns:
        The refresh plan: records to (re-)embed, record ids to delete, unchanged count.
    """
    to_embed: Final[list[IndexRecord]] = [
        record for record in records if stored.get(record.record_id) != record.content_hash
    ]
    current_ids: Final[set[str]] = {record.record_id for record in records}
    deleted_ids: Final[list[str]] = sorted(record_id for record_id in stored if record_id not in current_ids)
    return RefreshPlan(
        to_embed=tuple(to_embed),
        deleted_ids=tuple(deleted_ids),
        unchanged_count=len(records) - len(to_embed),
    )


@validate_call
def budgeted_embeds(
    to_embed: Sequence[IndexRecord], max_pdf_embeds: int | None
) -> tuple[list[IndexRecord], list[IndexRecord]]:
    """Split records due for embedding into those embedded now and those deferred.

    Page and block records are always embedded now — an edit must land.  PDF chunk records
    are embedded in order up to *max_pdf_embeds*; the rest are deferred, so they stay due
    and a later refresh picks them up.

    Args:
        to_embed: The records a refresh plan selected, in plan order.
        max_pdf_embeds: The most PDF chunk records to embed now; ``None`` for no cap.

    Returns:
        The records to embed now and the deferred ones, each in plan order.
    """
    if max_pdf_embeds is None:
        return list(to_embed), []
    now: Final[list[IndexRecord]] = []
    deferred: Final[list[IndexRecord]] = []
    pdf_count: int = 0
    for record in to_embed:
        if record.kind is not RecordKind.PDF:
            now.append(record)
        elif pdf_count < max_pdf_embeds:
            now.append(record)
            pdf_count += 1
        else:
            deferred.append(record)
    return now, deferred


@validate_call
def refresh_store(
    db_path: Path,
    api_endpoint: ApiEndpoint,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    include_daily_notes: bool = True,
    include_pdfs: bool = True,
    pdf_cache_path: Path | None = None,
    pdf_fetch_budget: PdfFetchBudget | None = None,
    max_pdf_embeds: int | None = None,
) -> RefreshSummary:
    """Plan and apply an incremental refresh against a store.

    Upgrades a schema-v2 store in place first.  Fetches and normalizes the whole graph,
    adds a record per chunk of every attached PDF the PDF text cache holds, embeds exactly
    the records the plan selects (with the store's own model), upserts them, deletes
    vanished records, and stamps the refresh moment and record count into the store's meta.
    With a *pdf_fetch_budget* it first fetches PDFs the cache lacks, within that budget;
    without one it fetches none, and a PDF the cache lacks simply has no records yet.

    Args:
        db_path: The store's database file.
        api_endpoint: The Local API endpoint for the store's graph.
        ollama_url: The local embedding server's base URL.
        include_daily_notes: Whether daily-note pages (and their blocks) are indexed;
            pass what the store was built with, or records will be added/deleted
            accordingly.
        include_pdfs: Whether attached PDFs' cached text is indexed.
        pdf_cache_path: The PDF text cache; defaults to the store's graph's conventional
            cache (:func:`~roam_semantic_search.pdf_text.default_pdf_cache_path`).
        pdf_fetch_budget: How much PDF fetching the refresh may do first; ``None`` fetches
            no PDFs.
        max_pdf_embeds: The most PDF chunk records to embed (:func:`budgeted_embeds`);
            ``None`` for no cap.

    Returns:
        The refresh summary.

    Raises:
        ValueError: If the store's schema version is not the current one (a full
            rebuild is required — an incremental refresh cannot migrate the layout).
    """
    upgrade_store(db_path)
    meta: Final[StoreMeta] = read_meta(db_path)
    if meta.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"store schema v{meta.schema_version} does not match current v{SCHEMA_VERSION}"
            " — rebuild the index with `roam-semantic-search build`"
        )
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    records: Final[list[IndexRecord]] = normalized_records(rows, include_daily_notes=include_daily_notes)
    extraction: ExtractionSummary | None = None
    pdf_error: str | None = None
    if include_pdfs:
        cache: Final[Path] = pdf_cache_path if pdf_cache_path is not None else default_pdf_cache_path(meta.graph_name)
        if pdf_fetch_budget is not None:
            try:
                extraction = extract_pdf_texts(
                    cache,
                    indexable_pdf_embeds(rows, include_daily_notes=include_daily_notes),
                    LocalApiGraphSource(api_endpoint=api_endpoint),
                    budget=pdf_fetch_budget,
                )
            except (RuntimeError, requests.exceptions.RequestException) as error:
                logger.warning("fetching PDFs failed; refreshing from the cached PDF text: %s", error)
                pdf_error = f"{type(error).__name__}: {error}"
        records.extend(pdf_chunk_records(rows, cached_pdf_texts(cache), include_daily_notes=include_daily_notes))
    plan: Final[RefreshPlan] = refresh_plan(records, stored_hashes(db_path))
    embed_now, deferred = budgeted_embeds(plan.to_embed, max_pdf_embeds)
    logger.info(
        "refresh plan: %d to embed (%d deferred), %d to delete, %d unchanged",
        len(embed_now),
        len(deferred),
        len(plan.deleted_ids),
        plan.unchanged_count,
    )

    if embed_now:
        vectors: Final[list[list[float]]] = embed_texts(
            [record.embed_input for record in embed_now], model=meta.embed_model, base_url=ollama_url
        )
        embeddings: Final[NDArray[np.float32]] = np.asarray(vectors, dtype=np.float32)
        if int(embeddings.shape[1]) != meta.dimension:
            raise ValueError(
                f"model {meta.embed_model!r} now answers dimension {int(embeddings.shape[1])},"
                f" but the store holds {meta.dimension} — rebuild the index"
            )
        upsert_records(db_path, embed_now, embeddings)
    if plan.deleted_ids:
        delete_records(db_path, list(plan.deleted_ids))

    record_count: Final[int] = stored_record_count(db_path)
    stamp_refresh(db_path, record_count)
    return RefreshSummary(
        embedded_count=len(embed_now),
        deleted_count=len(plan.deleted_ids),
        unchanged_count=plan.unchanged_count,
        record_count=record_count,
        pdf_fetched_count=extraction.extracted_count if extraction is not None else 0,
        pdf_pending_count=extraction.pending_count if extraction is not None else 0,
        embed_deferred_count=len(deferred),
        pdf_error=pdf_error,
    )
