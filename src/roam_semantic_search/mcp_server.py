"""The MCP server front end: the index's tools over stdio.

Exposes the index to MCP clients as three tools — ``semantic_search`` (hybrid
retrieval over the store), ``refresh_index`` (the incremental refresh), and
``index_stats`` (the store's meta).  Configuration arrives through environment
variables, since an MCP stdio server takes no CLI arguments of its own:

- ``GUFFIN_ROAM_GRAPH_NAME`` — the graph whose default store location is served.
- ``ROAM_SEMANTIC_SEARCH_DB`` — explicit store path, overriding the default location.
- ``ROAM_SEMANTIC_SEARCH_OLLAMA_URL`` — the local embedding server, when not the default.
- ``GUFFIN_ROAM_LOCAL_API_PORT`` / ``GUFFIN_ROAM_API_TOKEN`` — the Local API
  connection ``refresh_index`` uses (unneeded for pure search).

Tool functions are wired by the MCP server framework (their signatures become the
tool schemas), so they carry no ``@validate_call`` — the framework validates arguments.
"""

import logging
import os
from pathlib import Path
from typing import Final

from guffin.roam.local_api import ApiEndpoint
from mcp.server import MCPServer
from pydantic import BaseModel, ConfigDict

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL
from roam_semantic_search.query import SearchHit, search_store
from roam_semantic_search.refresh import RefreshSummary, refresh_store
from roam_semantic_search.store import StoreMeta, default_db_path, read_meta

logger = logging.getLogger(__name__)

server: Final[MCPServer] = MCPServer(
    "roam-semantic-search",
    instructions=(
        "Fully local semantic search over the user's Roam Research graph. Results carry Roam"
        " uids — reference a hit as ((uid)) in Roam contexts. The index is a snapshot of the"
        " graph: call refresh_index first when freshness matters (needs Roam Desktop running)."
    ),
)


class SearchResponse(BaseModel):
    """A ``semantic_search`` answer: the ranked hits plus the index's identity.

    Attributes:
        hits: The ranked search hits, best first.
        index: The store's provenance and embedding-model facts (including
            ``refreshed_at``, so a caller can judge staleness).
    """

    model_config = ConfigDict(frozen=True)

    hits: tuple[SearchHit, ...]
    index: StoreMeta


def _resolved_db_path() -> Path:
    """The store this server serves, from the environment.

    Returns:
        ``ROAM_SEMANTIC_SEARCH_DB`` when set, else the default location for
        ``GUFFIN_ROAM_GRAPH_NAME``.

    Raises:
        RuntimeError: If neither environment variable identifies a store, or the
            store file does not exist.
    """
    override: Final[str | None] = os.environ.get("ROAM_SEMANTIC_SEARCH_DB")
    graph_name: Final[str | None] = os.environ.get("GUFFIN_ROAM_GRAPH_NAME")
    if override:
        db_path = Path(override)
    elif graph_name:
        db_path = default_db_path(graph_name)
    else:
        raise RuntimeError("set ROAM_SEMANTIC_SEARCH_DB or GUFFIN_ROAM_GRAPH_NAME to identify the index")
    if not db_path.exists():
        raise RuntimeError(f"no index at {db_path} — run `roam-semantic-search build` first")
    return db_path


def _ollama_url() -> str:
    """The local embedding server's base URL, from the environment."""
    return os.environ.get("ROAM_SEMANTIC_SEARCH_OLLAMA_URL") or DEFAULT_OLLAMA_URL


@server.tool()
def semantic_search(query: str, k: int = 10) -> SearchResponse:
    """Search the Roam graph's local semantic index by meaning.

    Hybrid retrieval: the query is embedded locally and ranked against every
    record's vector, fused (reciprocal rank fusion) with a BM25 keyword ranking.
    Each hit carries the Roam uid — reference it as ``((uid))`` — plus its page,
    breadcrumb context path, and plain text.  The index is a snapshot: check the
    response's ``index.refreshed_at`` and call ``refresh_index`` first if staleness
    matters.

    Args:
        query: Natural-language query — paraphrase works; exact terms also rank.
        k: Maximum hits returned (default 10).

    Returns:
        The ranked hits plus the index's identity and freshness facts.
    """
    db_path: Final[Path] = _resolved_db_path()
    hits: Final[list[SearchHit]] = search_store(db_path, query, k=k, ollama_url=_ollama_url())
    return SearchResponse(hits=tuple(hits), index=read_meta(db_path))


@server.tool()
def refresh_index() -> RefreshSummary:
    """Incrementally refresh the index from the live Roam graph.

    Fetches the whole graph through the Roam Local API (Roam Desktop must be
    running), re-embeds only records whose content changed, and deletes records
    whose entities vanished.  Typically answers in a few seconds; the no-change
    case is around two.

    Returns:
        Counts of what the refresh embedded, deleted, and left untouched.
    """
    port: Final[str | None] = os.environ.get("GUFFIN_ROAM_LOCAL_API_PORT")
    graph_name: Final[str | None] = os.environ.get("GUFFIN_ROAM_GRAPH_NAME")
    token: Final[str | None] = os.environ.get("GUFFIN_ROAM_API_TOKEN")
    if not port or not graph_name or not token:
        raise RuntimeError(
            "refresh needs GUFFIN_ROAM_LOCAL_API_PORT, GUFFIN_ROAM_GRAPH_NAME, and GUFFIN_ROAM_API_TOKEN"
        )
    api_endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(
        local_api_port=int(port), graph_name=graph_name, bearer_token=token
    )
    return refresh_store(_resolved_db_path(), api_endpoint, ollama_url=_ollama_url())


@server.tool()
def index_stats() -> StoreMeta:
    """The index's provenance: graph, embedding model, record count, build/refresh moments.

    Returns:
        The store's meta facts.
    """
    return read_meta(_resolved_db_path())


def main() -> None:
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server.run()


if __name__ == "__main__":
    main()
