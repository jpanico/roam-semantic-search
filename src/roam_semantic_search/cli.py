"""The roam-semantic-search command-line interface.

Commands:

- ``graphs`` — list the graphs connected on this machine and which of them are indexed.
- ``build`` — fetch the graph, normalize, embed locally, and (re)write the index store.
- ``refresh`` — incrementally update the store: re-embed only changed records, delete vanished ones.
- ``extract-pdfs`` — fetch the graph's attached PDFs and SVGs and cache their text, for indexing.
- ``search`` — answer a natural-language query from the store.
- ``stats`` — print the store's provenance and embedding-model facts.

``--graph`` takes either the nickname a graph was connected under or its canonical Roam
name.  A nickname resolves through Roam's own config files
(:mod:`roam_semantic_search.graph_registry`) to the canonical name, the shared Local API
port, and that graph's bearer token, so ``--port`` and ``--token`` are needed only to
override them or to reach a graph the registry does not know.

Reaching a graph opens it: a Local API request naming a graph Roam Desktop cannot serve
makes Roam open a window for it, which for an encrypted graph is an unlock prompt.  An
unattended ``refresh`` should therefore pass ``--require-open-window``, which skips the
run — successfully, so a scheduler sees no failure — unless Roam's window-restore list
names the graph.  That test only rules out the graphs Roam certainly cannot serve; being
listed is no guarantee of being served, so the flag reduces prompts without abolishing
them (see :mod:`roam_semantic_search.graph_windows`).
"""

import logging
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import requests
import typer
from guffin.roam.local_api import ApiEndpoint
from guffin.roam.local_api_source import LocalApiGraphSource
from numpy.typing import NDArray

from roam_semantic_search.embed import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, embed_texts
from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.graph_registry import RegisteredGraph, local_api_port, registered_graphs, resolve_graph
from roam_semantic_search.graph_windows import WindowState, window_state_for
from roam_semantic_search.normalize import (
    IndexRecord,
    RecordKind,
    attachment_records,
    indexable_attachment_embeds,
    normalized_records,
)
from roam_semantic_search.pdf_text import (
    ExtractionSummary,
    PdfEmbed,
    PdfFetchBudget,
    cached_pdf_texts,
    default_pdf_cache_path,
    extract_pdf_texts,
    pdf_embeds,
    svg_embeds,
)
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
PdfCacheOption = Annotated[
    Path | None,
    typer.Option(
        "--pdf-cache", help="PDF text cache file; defaults to ~/.cache/roam-semantic-search/<graph>.pdf-text.db"
    ),
]
IncludePdfsOption = Annotated[
    bool, typer.Option("--pdfs/--no-pdfs", help="Index attached PDFs' text, fetching new ones within the budget")
]
MaxPdfsOption = Annotated[int | None, typer.Option("--max-pdfs", min=0, help="Fetch at most this many PDFs")]
MaxPdfMbOption = Annotated[
    float,
    typer.Option(
        "--max-pdf-mb",
        min=0,
        help="Stop fetching PDFs once this many MB have crossed Roam Desktop (a large run once crashed it)",
    ),
]
OcrOption = Annotated[
    bool,
    typer.Option(
        "--ocr/--no-ocr",
        help="Read scanned PDF pages by on-device OCR (macOS Vision; about a second a page)",
    ),
]
DEFAULT_MAX_PDF_MB: Final[float] = 500.0
"""A CLI run's default PDF byte budget: well under the ~1.3 GB after which a run crashed Roam Desktop."""
RequireOpenWindowOption = Annotated[
    bool,
    typer.Option(
        "--require-open-window/--no-require-open-window",
        envvar="ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW",
        help="Skip the run unless Roam Desktop's window-restore list names the graph",
    ),
]

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


def _pdf_cache_path(selector: str, cache_path: Path | None) -> Path:
    """The PDF text cache to use: an explicit ``--pdf-cache``, else the graph's default location."""
    return cache_path if cache_path is not None else default_pdf_cache_path(_canonical_graph_name(selector))


def _fetch_budget(max_pdfs: int | None, max_pdf_mb: float) -> PdfFetchBudget:
    """The PDF fetch budget the ``--max-pdfs`` / ``--max-pdf-mb`` options describe."""
    return PdfFetchBudget(max_pdfs=max_pdfs, max_bytes=int(max_pdf_mb * 1_000_000))


def _outcomes(summary: ExtractionSummary) -> str:
    """An extraction's outcomes as ``"3 no-text, 12 ok"``, or ``"none"``."""
    return ", ".join(f"{count} {status}" for status, count in sorted(summary.outcomes.items())) or "none"


def _fetch_pdfs(
    cache_path: Path, embeds: Sequence[PdfEmbed], api_endpoint: ApiEndpoint, budget: PdfFetchBudget, ocr: bool
) -> ExtractionSummary | None:
    """Fetch what the cache lacks within *budget*, reporting a failure rather than raising it."""
    try:
        return extract_pdf_texts(
            cache_path, embeds, LocalApiGraphSource(api_endpoint=api_endpoint), budget=budget, ocr=ocr
        )
    except (RuntimeError, requests.exceptions.RequestException) as error:
        typer.echo(f"fetching PDFs failed; indexing the cached PDF text only: {error}", err=True)
        return None


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
    include_pdfs: IncludePdfsOption = True,
    pdf_cache_path: PdfCacheOption = None,
    max_pdfs: MaxPdfsOption = None,
    max_pdf_mb: MaxPdfMbOption = DEFAULT_MAX_PDF_MB,
    ocr: OcrOption = True,
) -> None:
    """Fetch the graph, normalize, embed locally, and (re)write the index store.

    Attached PDFs the PDF text cache lacks are fetched first, within ``--max-pdfs`` and
    ``--max-pdf-mb``; any left over are indexed by a later ``refresh`` or ``extract-pdfs``.
    """
    api_endpoint: Final[ApiEndpoint] = _api_endpoint(graph, port, token)
    resolved_db: Final[Path] = _store_path(graph, db_path)

    started: Final[float] = time.perf_counter()
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    records: Final[list[IndexRecord]] = normalized_records(rows, include_daily_notes=include_daily_notes)
    if include_pdfs:
        cache: Final[Path] = _pdf_cache_path(graph, pdf_cache_path)
        extraction: Final[ExtractionSummary | None] = _fetch_pdfs(
            cache,
            indexable_attachment_embeds(rows, include_daily_notes=include_daily_notes),
            api_endpoint,
            _fetch_budget(max_pdfs, max_pdf_mb),
            ocr,
        )
        if extraction is not None:
            typer.echo(
                f"PDFs: fetched {extraction.extracted_count} ({_outcomes(extraction)};"
                f" {extraction.ocr_page_count} pages by OCR), {extraction.pending_count} pending"
            )
        records.extend(attachment_records(rows, cached_pdf_texts(cache), include_daily_notes=include_daily_notes))
    pdf_count: Final[int] = sum(1 for record in records if record.kind is RecordKind.PDF)
    typer.echo(f"fetched {len(rows)} entities; {len(records)} indexable records ({pdf_count} PDF chunks)")

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


def _open_window_refused(selector: str) -> bool:
    """Whether an open-window requirement forbids touching *selector* right now.

    Reports the reason it does, so a skipped scheduled run is legible in the log rather
    than merely silent.

    Args:
        selector: The graph a command was asked to operate on.

    Returns:
        ``True`` when Roam's window-restore list does not name the graph — or cannot be
        read — and the run must therefore be skipped.  A ``False`` here clears the run
        to proceed; it does not predict that the request will be served.
    """
    graph_name: Final[str] = _canonical_graph_name(selector)
    state: Final[WindowState] = window_state_for(graph_name)
    if state is WindowState.OPEN:
        return False
    if state is WindowState.CLOSED:
        typer.echo(f"skipped: Roam Desktop has no window open on {graph_name}")
        return True
    typer.echo(f"skipped: cannot tell whether Roam Desktop has a window open on {graph_name}", err=True)
    return True


@app.command()
def refresh(
    graph: GraphOption,
    port: PortOption = None,
    token: TokenOption = None,
    db_path: DbOption = None,
    ollama_url: OllamaUrlOption = DEFAULT_OLLAMA_URL,
    include_daily_notes: Annotated[bool, typer.Option("--daily-notes/--no-daily-notes")] = True,
    include_pdfs: IncludePdfsOption = True,
    pdf_cache_path: PdfCacheOption = None,
    max_pdfs: MaxPdfsOption = None,
    max_pdf_mb: MaxPdfMbOption = DEFAULT_MAX_PDF_MB,
    max_pdf_embeds: Annotated[
        int | None, typer.Option("--max-pdf-embeds", min=0, help="Embed at most this many PDF passages")
    ] = None,
    ocr: OcrOption = True,
    require_open_window: RequireOpenWindowOption = False,
) -> None:
    """Incrementally update the store: re-embed only changed records, delete vanished ones.

    Attached PDFs the PDF text cache lacks are fetched first, within ``--max-pdfs`` and
    ``--max-pdf-mb``; ``--max-pdf-embeds`` caps how many PDF passages are embedded, the rest
    waiting for the next refresh (page and block edits are never held back).  A schema-v2
    store is upgraded in place first, embeddings kept.

    Fetching reaches the graph through the Roam Local API, and a request naming a graph Roam
    Desktop cannot serve makes it open a window — an unlock prompt, for an encrypted graph.
    ``--require-open-window`` skips the run, successfully, unless Roam's window-restore list
    names the graph.  That rules out the certainly-unservable graphs, not every prompt: a
    listed graph may still be closed or locked, and only the request itself finds out.
    """
    resolved_db: Final[Path] = _store_path(graph, db_path)
    if not resolved_db.exists():
        typer.echo(f"no index at {resolved_db} — run `roam-semantic-search build` first", err=True)
        raise typer.Exit(code=1)
    if require_open_window and _open_window_refused(graph):
        return
    api_endpoint: Final[ApiEndpoint] = _api_endpoint(graph, port, token)
    started: Final[float] = time.perf_counter()
    summary: Final[RefreshSummary] = refresh_store(
        resolved_db,
        api_endpoint,
        ollama_url=ollama_url,
        include_daily_notes=include_daily_notes,
        include_pdfs=include_pdfs,
        pdf_cache_path=_pdf_cache_path(graph, pdf_cache_path),
        pdf_fetch_budget=_fetch_budget(max_pdfs, max_pdf_mb),
        max_pdf_embeds=max_pdf_embeds,
        ocr=ocr,
    )
    elapsed: Final[float] = time.perf_counter() - started
    typer.echo(
        f"refreshed {resolved_db} in {elapsed:.1f}s: {summary.embedded_count} re-embedded,"
        f" {summary.deleted_count} deleted, {summary.unchanged_count} unchanged"
        f" ({summary.record_count} records)"
    )
    if include_pdfs:
        typer.echo(
            f"PDFs: {summary.pdf_fetched_count} fetched, {summary.pdf_pending_count} still to fetch,"
            f" {summary.embed_deferred_count} passages left to embed"
        )
    if summary.pdf_error is not None:
        typer.echo(f"fetching PDFs failed; indexed the cached PDF text only: {summary.pdf_error}", err=True)


@app.command("extract-pdfs")
def extract_pdfs(
    graph: GraphOption,
    port: PortOption = None,
    token: TokenOption = None,
    pdf_cache_path: PdfCacheOption = None,
    retry_failed: Annotated[
        bool, typer.Option("--retry-failed", help="Refetch PDFs whose fetch or parse failed before")
    ] = False,
    max_pdfs: MaxPdfsOption = None,
    max_pdf_mb: MaxPdfMbOption = DEFAULT_MAX_PDF_MB,
    ocr: OcrOption = True,
    require_open_window: RequireOpenWindowOption = False,
) -> None:
    """Fetch the graph's attached PDFs and SVGs and cache their text, for indexing.

    Every PDF a block embeds is fetched decrypted through the Roam Local API, held in memory
    only, and its text layer cached page by page in an owner-only file; already-cached PDFs
    are skipped, so a run picks up where the last one stopped, and ``--max-pdfs`` /
    ``--max-pdf-mb`` bound one run.  Reaching the graph can make
    Roam Desktop open a window for it — see ``refresh`` for ``--require-open-window``.
    """
    if require_open_window and _open_window_refused(graph):
        return
    api_endpoint: Final[ApiEndpoint] = _api_endpoint(graph, port, token)
    resolved_cache: Final[Path] = _pdf_cache_path(graph, pdf_cache_path)
    started: Final[float] = time.perf_counter()
    rows: Final[list[dict[str, object]]] = fetch_graph(api_endpoint)
    embeds: Final[list[PdfEmbed]] = [*pdf_embeds(rows), *svg_embeds(rows)]
    try:
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            resolved_cache,
            embeds,
            LocalApiGraphSource(api_endpoint=api_endpoint),
            retry_failed=retry_failed,
            budget=_fetch_budget(max_pdfs, max_pdf_mb),
            ocr=ocr,
        )
    except RuntimeError as error:
        typer.echo(f"aborted: {error}", err=True)
        raise typer.Exit(code=1) from error
    elapsed: Final[float] = time.perf_counter() - started
    typer.echo(
        f"{resolved_cache}: {summary.distinct_count} distinct attachments in {len(embeds)} embeds;"
        f" extracted {summary.extracted_count} ({summary.fetched_bytes / 1_000_000:.0f} MB) in {elapsed:.1f}s"
        f" ({_outcomes(summary)}; {summary.ocr_page_count} pages by OCR); {summary.current_count} already current,"
        f" {summary.pending_count} pending"
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
        kind: str = f"pdf p. {hit.pdf_page}" if hit.kind is RecordKind.PDF else str(hit.kind)
        # An image hit's breadcrumb ends with its upload name, as a PDF hit's does.
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
