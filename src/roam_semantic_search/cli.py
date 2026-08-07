"""The roam-semantic-search command-line interface.

Commands:

- ``graphs`` — list the graphs connected on this machine and which of them are indexed.
- ``build`` — fetch the graph, normalize, embed locally, and (re)write the index store.
- ``refresh`` — incrementally update the store: re-embed only changed records, delete vanished ones.
- ``search`` — answer a natural-language query from the store.
- ``stats`` — print the store's provenance and embedding-model facts.

``--graph`` takes either the nickname a graph was connected under or its canonical Roam
name.  A nickname resolves through Roam's own config files
(:mod:`roam_semantic_search.graph_registry`) to the canonical name, the shared Local API
port, and that graph's bearer token, so ``--port`` and ``--token`` are needed only to
override them or to reach a graph the registry does not know.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import typer
from guffin.roam.local_api import ApiEndpoint
from numpy.typing import NDArray

from roam_semantic_search.embed import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, embed_texts
from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.graph_registry import RegisteredGraph, local_api_port, registered_graphs, resolve_graph
from roam_semantic_search.normalize import IndexRecord, normalized_records
from roam_semantic_search.query import SearchHit, search_store
from roam_semantic_search.refresh import RefreshSummary, refresh_store
from roam_semantic_search.store import StoreMeta, default_db_path, read_meta, write_store

app: Final[typer.Typer] = typer.Typer(add_completion=False, no_args_is_help=True)

# --port and --token carry no envvar= on purpose: Typer would fill the parameter from the
# environment and make it indistinguishable from an explicit flag, letting the environment's
# *default-graph* credential outrank the registry entry of whatever graph was actually named
# (a 401 in practice).  The documented precedence — flag, then registry, then environment —
# therefore reads the environment explicitly, last, in _api_endpoint.
PortOption = Annotated[int | None, typer.Option("--port", "-p")]
GraphOption = Annotated[
    str, typer.Option("--graph", "-g", envvar="GUFFIN_ROAM_GRAPH_NAME", help="Graph nickname or canonical name")
]
TokenOption = Annotated[str | None, typer.Option("--token", "-t")]
DbOption = Annotated[
    Path | None, typer.Option("--db", help="Index database file; defaults to ~/.cache/roam-semantic-search/<graph>.db")
]
OllamaUrlOption = Annotated[str, typer.Option("--ollama-url", envvar="ROAM_SEMANTIC_SEARCH_OLLAMA_URL")]

_TEXT_PREVIEW_MAX_CHARS: Final[int] = 200


def _registered(selector: str) -> RegisteredGraph | None:
    """The registry entry *selector* names, or ``None`` when the registry does not know it."""
    try:
        return resolve_graph(selector)
    except RuntimeError:
        return None


def _canonical_graph_name(selector: str) -> str:
    """The canonical Roam name *selector* denotes, falling back to *selector* itself.

    Store locations key off the canonical name, never the nickname, so the CLI and the MCP
    server address the same file for the same graph.
    """
    entry: Final[RegisteredGraph | None] = _registered(selector)
    return entry.name if entry else selector


def _store_path(selector: str, db_path: Path | None) -> Path:
    """The store to operate on: an explicit ``--db``, else the graph's default location."""
    return db_path if db_path is not None else default_db_path(_canonical_graph_name(selector))


def _api_endpoint(selector: str, port: int | None, token: str | None) -> ApiEndpoint:
    """Build the Local API endpoint for *selector*.

    An explicit flag wins; otherwise the registry supplies the graph's own token and the
    shared port, and only then does the environment fill any remaining gap.  Preferring the
    registry over the environment matters: the environment names one default graph, and its
    token is the wrong credential for any other.
    """
    entry: Final[RegisteredGraph | None] = _registered(selector)
    resolved_token: Final[str | None] = (
        token or (entry.token if entry else None) or os.environ.get("GUFFIN_ROAM_API_TOKEN")
    )
    if not resolved_token:
        typer.echo(f"no token for graph {selector!r}: pass --token, or connect the graph in Roam", err=True)
        raise typer.Exit(code=1)

    resolved_port: int | None = port
    if resolved_port is None:
        try:
            resolved_port = local_api_port()
        except RuntimeError:
            env_port: Final[str | None] = os.environ.get("GUFFIN_ROAM_LOCAL_API_PORT")
            resolved_port = int(env_port) if env_port else None
    if resolved_port is None:
        typer.echo("no Local API port: pass --port, or run Roam Desktop once to record it", err=True)
        raise typer.Exit(code=1)

    return ApiEndpoint.from_parts(
        local_api_port=resolved_port,
        graph_name=_canonical_graph_name(selector),
        bearer_token=resolved_token,
    )


@app.callback()
def configure() -> None:
    """Configure logging for every command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def graphs() -> None:
    """List the graphs connected on this machine and which of them are indexed."""
    entries: Final[tuple[RegisteredGraph, ...]] = registered_graphs()
    if not entries:
        typer.echo("no graphs connected — connect one in Roam Desktop first")
        return
    for entry in entries:
        store: Path = default_db_path(entry.name)
        state: str = f"indexed ({read_meta(store).record_count} records)" if store.exists() else "not indexed"
        typer.echo(f"{entry.nickname or entry.name:<12} {entry.name:<12} {entry.graph_type:<8} {state}")


@app.command()
def build(
    graph: GraphOption,
    port: PortOption = None,
    token: TokenOption = None,
    db_path: DbOption = None,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBED_MODEL,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    include_daily_notes: Annotated[bool, typer.Option("--daily-notes/--no-daily-notes")] = True,
) -> None:
    """Fetch the graph, normalize, embed locally, and (re)write the index store."""
    api_endpoint: Final[ApiEndpoint] = _api_endpoint(graph, port, token)
    resolved_db: Final[Path] = _store_path(graph, db_path)

    started: Final[float] = time.perf_counter()
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    records: Final[list[IndexRecord]] = normalized_records(rows, include_daily_notes=include_daily_notes)
    typer.echo(f"fetched {len(rows)} entities; {len(records)} indexable records")

    vectors: Final[list[list[float]]] = embed_texts(
        [record.embed_input for record in records], model=model, base_url=ollama_url
    )
    embeddings: Final[NDArray[np.float32]] = np.asarray(vectors, dtype=np.float32)
    meta: Final[StoreMeta] = StoreMeta(
        graph_name=_canonical_graph_name(graph),
        embed_model=model,
        dimension=int(embeddings.shape[1]),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        record_count=len(records),
    )
    write_store(resolved_db, records, embeddings, meta)
    elapsed: Final[float] = time.perf_counter() - started
    typer.echo(f"built {resolved_db} ({len(records)} records, dim {meta.dimension}) in {elapsed:.1f}s")


@app.command()
def refresh(
    graph: GraphOption,
    port: PortOption = None,
    token: TokenOption = None,
    db_path: DbOption = None,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    include_daily_notes: Annotated[bool, typer.Option("--daily-notes/--no-daily-notes")] = True,
) -> None:
    """Incrementally update the store: re-embed only changed records, delete vanished ones."""
    resolved_db: Final[Path] = _store_path(graph, db_path)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db} — run `roam-semantic-search build` first", err=True)
        raise typer.Exit(code=1)
    api_endpoint: Final[ApiEndpoint] = _api_endpoint(graph, port, token)
    started: Final[float] = time.perf_counter()
    summary: Final[RefreshSummary] = refresh_store(
        resolved_db, api_endpoint, ollama_url=ollama_url, include_daily_notes=include_daily_notes
    )
    elapsed: Final[float] = time.perf_counter() - started
    typer.echo(
        f"refreshed {resolved_db} in {elapsed:.1f}s: {summary.embedded_count} re-embedded,"
        f" {summary.deleted_count} deleted, {summary.unchanged_count} unchanged"
        f" ({summary.record_count} records)"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language query")],
    graph: GraphOption,
    db_path: DbOption = None,
    k: Annotated[int, typer.Option("--k", "-k")] = 10,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
) -> None:
    """Answer a natural-language query from the index store."""
    resolved_db: Final[Path] = _store_path(graph, db_path)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db} — run `roam-semantic-search build` first", err=True)
        raise typer.Exit(code=1)
    hits: Final[list[SearchHit]] = search_store(resolved_db, query, k=k, ollama_url=ollama_url)
    for rank, hit in enumerate(hits, start=1):
        kind: str = "page" if hit.is_page else "block"
        ranks: str = f"v:{hit.vector_rank or '-'} k:{hit.keyword_rank or '-'}"
        typer.echo(f"{rank:2}. (({hit.uid}))  [{kind}  score {hit.score:.4f}  {ranks}]")
        typer.echo(f"    {hit.breadcrumb}")
        if not hit.is_page:
            typer.echo(f"    {hit.text[:_TEXT_PREVIEW_MAX_CHARS]}")


@app.command()
def stats(
    graph: GraphOption,
    db_path: DbOption = None,
) -> None:
    """Print the store's provenance and embedding-model facts."""
    resolved_db: Final[Path] = _store_path(graph, db_path)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db}", err=True)
        raise typer.Exit(code=1)
    meta: Final[StoreMeta] = read_meta(resolved_db)
    for key, value in meta.model_dump().items():
        typer.echo(f"{key}: {value}")


if __name__ == "__main__":
    app()
