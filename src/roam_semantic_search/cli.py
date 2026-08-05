"""The roam-semantic-search command-line interface.

Commands:

- ``build`` — fetch the graph, normalize, embed locally, and (re)write the index store.
- ``refresh`` — incrementally update the store: re-embed only changed records, delete vanished ones.
- ``search`` — answer a natural-language query from the store.
- ``stats`` — print the store's provenance and embedding-model facts.
"""

import logging
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
from roam_semantic_search.normalize import IndexRecord, normalized_records
from roam_semantic_search.query import SearchHit, search_store
from roam_semantic_search.refresh import RefreshSummary, refresh_store
from roam_semantic_search.store import StoreMeta, default_db_path, read_meta, write_store

app: Final[typer.Typer] = typer.Typer(add_completion=False, no_args_is_help=True)

PortOption = Annotated[int, typer.Option("--port", "-p", envvar="GUFFIN_ROAM_LOCAL_API_PORT")]
GraphOption = Annotated[str, typer.Option("--graph", "-g", envvar="GUFFIN_ROAM_GRAPH_NAME")]
TokenOption = Annotated[str, typer.Option("--token", "-t", envvar="GUFFIN_ROAM_API_TOKEN")]
DbOption = Annotated[
    Path | None, typer.Option("--db", help="Index database file; defaults to ~/.cache/roam-semantic-search/<graph>.db")
]
OllamaUrlOption = Annotated[str, typer.Option("--ollama-url", envvar="ROAM_SEMANTIC_SEARCH_OLLAMA_URL")]

_TEXT_PREVIEW_MAX_CHARS: Final[int] = 200


@app.callback()
def configure() -> None:
    """Configure logging for every command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def build(
    port: PortOption,
    graph: GraphOption,
    token: TokenOption,
    db_path: DbOption = None,
    model: Annotated[str, typer.Option("--model")] = DEFAULT_EMBED_MODEL,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    include_daily_notes: Annotated[bool, typer.Option("--daily-notes/--no-daily-notes")] = False,
) -> None:
    """Fetch the graph, normalize, embed locally, and (re)write the index store."""
    api_endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(local_api_port=port, graph_name=graph, bearer_token=token)
    resolved_db: Final[Path] = db_path if db_path is not None else default_db_path(graph)

    started: Final[float] = time.perf_counter()
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    records: Final[list[IndexRecord]] = normalized_records(rows, include_daily_notes=include_daily_notes)
    typer.echo(f"fetched {len(rows)} entities; {len(records)} indexable records")

    vectors: Final[list[list[float]]] = embed_texts(
        [record.embed_input for record in records], model=model, base_url=ollama_url
    )
    embeddings: Final[NDArray[np.float32]] = np.asarray(vectors, dtype=np.float32)
    meta: Final[StoreMeta] = StoreMeta(
        graph_name=graph,
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
    port: PortOption,
    graph: GraphOption,
    token: TokenOption,
    db_path: DbOption = None,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    include_daily_notes: Annotated[bool, typer.Option("--daily-notes/--no-daily-notes")] = False,
) -> None:
    """Incrementally update the store: re-embed only changed records, delete vanished ones."""
    resolved_db: Final[Path] = db_path if db_path is not None else default_db_path(graph)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db} — run `roam-semantic-search build` first", err=True)
        raise typer.Exit(code=1)
    api_endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(local_api_port=port, graph_name=graph, bearer_token=token)
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
    resolved_db: Final[Path] = db_path if db_path is not None else default_db_path(graph)
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
    resolved_db: Final[Path] = db_path if db_path is not None else default_db_path(graph)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db}", err=True)
        raise typer.Exit(code=1)
    meta: Final[StoreMeta] = read_meta(resolved_db)
    for key, value in meta.model_dump().items():
        typer.echo(f"{key}: {value}")


if __name__ == "__main__":
    app()
