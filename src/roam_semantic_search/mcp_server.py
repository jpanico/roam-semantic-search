"""The MCP server front end: the index's tools over stdio.

Exposes the index to MCP clients as four tools — ``list_indexes`` (the graphs that can be
searched), ``semantic_search`` (hybrid retrieval over a store), ``refresh_index`` (the
incremental refresh), and ``index_stats`` (a store's meta).

Every graph-touching tool takes a required ``graph`` argument, naming a graph by the
nickname it was connected under.  Requiring it is deliberate: several graphs are typically
connected at once, and a server-wide default would silently answer from whichever graph the
process happened to be configured for — a wrong answer that reads exactly like a right one.
``list_indexes`` exists so a caller can discover the legal values rather than guess.

Graph identity, the shared Local API port, and per-graph tokens all come from Roam's own
config files (see :mod:`roam_semantic_search.graph_registry`), so this server needs no
graph configuration of its own.  The remaining environment variables are:

- ``ROAM_SEMANTIC_SEARCH_OLLAMA_URL`` — the local embedding server, when not the default.
- ``ROAM_SEMANTIC_SEARCH_MAX_STALENESS`` — the auto-refresh staleness threshold, in
  seconds (default 3600); a negative value disables auto-refresh entirely.

A search answers from the index — a snapshot — never from the live graph.  To bound how
stale an answer can be, ``semantic_search`` first auto-refreshes the index when its last
capture is older than the staleness threshold; a failed refresh (most commonly Roam
Desktop not running) degrades gracefully to the existing snapshot, and every response
reports what happened in its ``refresh`` field so staleness is visible, never silent.

Tool functions are wired by the MCP server framework (their signatures become the
tool schemas), so they carry no ``@validate_call`` — the framework validates arguments.
"""

import enum
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL
from roam_semantic_search.graph_registry import RegisteredGraph, api_endpoint_for, registered_graphs, resolve_graph
from roam_semantic_search.query import SearchHit, search_store
from roam_semantic_search.refresh import RefreshSummary, refresh_store
from roam_semantic_search.store import StoreMeta, default_db_path, read_meta

logger = logging.getLogger(__name__)

server: Final[MCPServer] = MCPServer(
    "roam-semantic-search",
    instructions=(
        "Fully local semantic search over the user's Roam Research graphs. Several graphs may be"
        " connected, so every tool requires a graph argument (its nickname) — call list_indexes"
        " first to see which graphs are connected and which have a searchable index. Results carry"
        " Roam uids — reference a hit as ((uid)) in Roam contexts. An index is a snapshot of its"
        " graph: call refresh_index first when freshness matters (needs Roam Desktop running)."
    ),
)


DEFAULT_MAX_STALENESS_SECONDS: Final[int] = 3600


class RefreshOutcome(enum.StrEnum):
    """What the pre-search staleness check did."""

    FRESH = "fresh"
    REFRESHED = "refreshed"
    REFRESH_FAILED = "refresh-failed"
    DISABLED = "disabled"


class AutoRefresh(BaseModel):
    """The pre-search staleness check's outcome, reported so staleness is never silent.

    Attributes:
        outcome: What the check did.
        age_seconds: How old the index's last capture was when checked, or ``None``
            when the store carries no parseable capture moment (treated as stale).
        threshold_seconds: The staleness threshold the age was judged against.
        error: Why a refresh failed, when one did — most commonly Roam Desktop not
            running; the search then answered from the existing snapshot.
    """

    model_config = ConfigDict(frozen=True)

    outcome: RefreshOutcome
    age_seconds: float | None
    threshold_seconds: int
    error: str | None = None


class SearchResponse(BaseModel):
    """A ``semantic_search`` answer: the ranked hits plus the index's identity.

    Attributes:
        hits: The ranked search hits, best first.
        index: The store's provenance and embedding-model facts (including
            ``refreshed_at``, so a caller can judge staleness).
        refresh: What the pre-search staleness check did — whether these hits come
            from a just-refreshed index or an aging snapshot.
    """

    model_config = ConfigDict(frozen=True)

    hits: tuple[SearchHit, ...]
    index: StoreMeta
    refresh: AutoRefresh


class IndexSummary(BaseModel):
    """One connected graph and the state of its index.

    Attributes:
        nickname: The label to pass as a tool's ``graph`` argument.
        name: The canonical Roam graph name.
        graph_type: Whether the graph's storage is cloud-hosted or local-only.
        indexed: Whether a built index exists for the graph.
        db_path: Where that index lives, whether or not it exists yet.
    """

    model_config = ConfigDict(frozen=True)

    nickname: str
    name: str
    graph_type: str
    indexed: bool
    db_path: str


def _ollama_url() -> str:
    """The local embedding server's base URL, from the environment."""
    return os.environ.get("ROAM_SEMANTIC_SEARCH_OLLAMA_URL") or DEFAULT_OLLAMA_URL


def _max_staleness_seconds() -> int:
    """The auto-refresh staleness threshold, from the environment; negative disables."""
    raw: Final[str | None] = os.environ.get("ROAM_SEMANTIC_SEARCH_MAX_STALENESS")
    if not raw:
        return DEFAULT_MAX_STALENESS_SECONDS
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer ROAM_SEMANTIC_SEARCH_MAX_STALENESS=%r", raw)
        return DEFAULT_MAX_STALENESS_SECONDS


@validate_call
def index_age_seconds(meta: StoreMeta, now: datetime) -> float | None:
    """How old a store's last capture is, judged at *now*.

    The capture moment is the latest refresh when one has run, else the build.  A naive
    stored timestamp is taken as UTC.

    Args:
        meta: The store's meta facts.
        now: The moment to judge age at; must be timezone-aware.

    Returns:
        The age in seconds (negative if the capture claims to be in the future), or
        ``None`` when the store carries no parseable capture moment.
    """
    stamp: Final[str | None] = meta.refreshed_at or meta.built_at
    if not stamp:
        return None
    try:
        captured: datetime = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=UTC)
    return (now - captured).total_seconds()


def _ensured_fresh(graph: str, db_path: Path) -> AutoRefresh:
    """Refresh *graph*'s store when its snapshot is older than the staleness threshold.

    A failed refresh is an outcome, not an error: search must keep answering from the
    existing snapshot when Roam Desktop is closed, so the failure is reported in the
    returned outcome rather than raised.
    """
    threshold: Final[int] = _max_staleness_seconds()
    if threshold < 0:
        return AutoRefresh(outcome=RefreshOutcome.DISABLED, age_seconds=None, threshold_seconds=threshold)

    age: Final[float | None] = index_age_seconds(read_meta(db_path), datetime.now(UTC))
    if age is not None and age <= threshold:
        return AutoRefresh(outcome=RefreshOutcome.FRESH, age_seconds=age, threshold_seconds=threshold)

    try:
        refresh_store(db_path, api_endpoint_for(resolve_graph(graph)), ollama_url=_ollama_url())
    except Exception as exc:  # noqa: BLE001 — degrade to the snapshot, whatever the cause
        logger.warning("auto-refresh of %r failed; answering from the snapshot: %s", graph, exc)
        return AutoRefresh(
            outcome=RefreshOutcome.REFRESH_FAILED, age_seconds=age, threshold_seconds=threshold, error=str(exc)
        )
    return AutoRefresh(outcome=RefreshOutcome.REFRESHED, age_seconds=age, threshold_seconds=threshold)


def _store_for(graph: str) -> Path:
    """The index store of the graph *graph* names.

    Args:
        graph: A connected graph's nickname or canonical name.

    Returns:
        The store's path.

    Raises:
        RuntimeError: If no such graph is connected, or it has no built index.
    """
    resolved: Final[RegisteredGraph] = resolve_graph(graph)
    db_path: Final[Path] = default_db_path(resolved.name)
    if not db_path.exists():
        raise RuntimeError(
            f"graph {graph!r} has no index at {db_path} — build one with "
            f"`roam-semantic-search build --graph {resolved.name}`"
        )
    return db_path


@server.tool()
def list_indexes() -> tuple[IndexSummary, ...]:
    """The graphs connected on this machine, and which of them can be searched.

    Call this before the other tools: they each require a ``graph`` argument, and this
    reports the nicknames that are legal values along with whether each graph has been
    indexed yet.

    Returns:
        One summary per connected graph, in registry order.
    """
    return tuple(
        IndexSummary(
            nickname=graph.nickname or graph.name,
            name=graph.name,
            graph_type=str(graph.graph_type),
            indexed=default_db_path(graph.name).exists(),
            db_path=str(default_db_path(graph.name)),
        )
        for graph in registered_graphs()
    )


@server.tool()
def semantic_search(graph: str, query: str, k: int = 10) -> SearchResponse:
    """Search one Roam graph's local semantic index by meaning.

    Hybrid retrieval: the query is embedded locally and ranked against every
    record's vector, fused (reciprocal rank fusion) with a BM25 keyword ranking.
    Each hit carries the Roam uid — reference it as ``((uid))`` — plus its page,
    breadcrumb context path, and plain text.

    The index is a snapshot, but staleness is bounded: when the last capture is older
    than the staleness threshold (``ROAM_SEMANTIC_SEARCH_MAX_STALENESS``, default one
    hour) the index is refreshed before searching.  A failed refresh — most commonly
    Roam Desktop not running — degrades gracefully to the snapshot; the response's
    ``refresh`` field says which of these happened, so check it when currency matters.

    Args:
        graph: Which graph to search, by nickname — see ``list_indexes``.
        query: Natural-language query — paraphrase works; exact terms also rank.
        k: Maximum hits returned (default 10).

    Returns:
        The ranked hits, the index's identity and freshness facts, and what the
        pre-search staleness check did.
    """
    db_path: Final[Path] = _store_for(graph)
    refresh: Final[AutoRefresh] = _ensured_fresh(graph, db_path)
    hits: Final[list[SearchHit]] = search_store(db_path, query, k=k, ollama_url=_ollama_url())
    return SearchResponse(hits=tuple(hits), index=read_meta(db_path), refresh=refresh)


@server.tool()
def refresh_index(graph: str) -> RefreshSummary:
    """Incrementally refresh one graph's index from the live Roam graph.

    Fetches the whole graph through the Roam Local API (Roam Desktop must be
    running), re-embeds only records whose content changed, and deletes records
    whose entities vanished.  Typically answers in a few seconds; the no-change
    case is around two.

    Args:
        graph: Which graph to refresh, by nickname — see ``list_indexes``.

    Returns:
        Counts of what the refresh embedded, deleted, and left untouched.
    """
    resolved: Final[RegisteredGraph] = resolve_graph(graph)
    return refresh_store(_store_for(graph), api_endpoint_for(resolved), ollama_url=_ollama_url())


@server.tool()
def index_stats(graph: str) -> StoreMeta:
    """One index's provenance: graph, embedding model, record count, build/refresh moments.

    Args:
        graph: Which graph's index to report on, by nickname — see ``list_indexes``.

    Returns:
        The store's meta facts.
    """
    return read_meta(_store_for(graph))


def main() -> None:
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server.run()


if __name__ == "__main__":
    main()
