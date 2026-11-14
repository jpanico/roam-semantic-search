"""PDF attachment text: find a graph's PDF embeds, extract their text, and cache it.

A PDF sits in a block as ``{{pdf: <url>}}`` or ``{{[[pdf]]: <url>}}``; the full-graph pull
carries only that markup, never the file.  This module reaches the content: each distinct URL
is fetched through the Local API's ``file.get`` action (which returns an encrypted graph's
asset decrypted), held in memory only, its text layer extracted page by page with pypdfium2,
and the result cached by URL.  A Firebase Storage URL names one immutable upload, so a cached
entry never goes stale; only a newer extractor (:data:`EXTRACTOR_VERSION`) re-extracts it.

The cache is its own SQLite file beside the index store, because ``build`` deletes and
recreates the store and a rebuild must not refetch every PDF.  It holds decrypted text, so it
is kept owner-only (``0600``).  No PDF bytes are ever written to disk, and the cache keys
entries by a digest of the URL, never the URL itself (a Firebase URL embeds an access token).

Public symbols:

- :class:`PdfEmbed` — one PDF embed: the hosting block's uid and the PDF's storage URL.
- :class:`PdfStatus` — the outcome of one PDF's extraction.
- :class:`PdfText` — one cached extraction result.
- :class:`PdfFetchBudget` — how much one run may fetch: PDFs, bytes, seconds.
- :class:`ExtractionSummary` — what an extraction run did, by count.
- :data:`EXTRACTOR_VERSION` — the extraction logic's version; a bump re-extracts.
- :data:`NO_TEXT_PAGE_MIN_CHARS` — the non-whitespace chars a page needs to count as text.
- :data:`CONSECUTIVE_FETCH_FAILURE_LIMIT` — failed fetches in a row that abort a run.
- :func:`default_pdf_cache_path` — the conventional cache location for a graph.
- :func:`pdf_embeds` — pull-block rows → every PDF embed, in row order.
- :func:`url_key` — the cache key of a PDF URL.
- :func:`extracted_text` — PDF bytes → status, per-page text, error.
- :func:`cached_pdf_texts` — every cached extraction result by URL key.
- :func:`extract_pdf_texts` — fetch and extract every embed the cache lacks.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# Rationale: pypdfium2 ships no type stubs, so some of its members are partially Unknown to
# pyright; both rules fire only on that propagation (guffin's pdf_raster.py does the same).

import enum
import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pypdfium2
import pypdfium2.raw as pdfium_raw
import regex
import requests
from guffin.roam.asset import RoamAsset
from guffin.roam.graph_source import GraphSource
from guffin.roam.markdown import PDF_EMBED_RE
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, validate_call

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION: Final[int] = 1
"""The extraction logic's version; entries extracted by an older one are re-extracted."""

NO_TEXT_PAGE_MIN_CHARS: Final[int] = 20
"""Non-whitespace text-layer chars a page needs to count as carrying text (fewer: a scan)."""

CONSECUTIVE_FETCH_FAILURE_LIMIT: Final[int] = 5
"""Failed fetches in a row that abort a run: the graph itself, not the files, is failing."""

_CACHE_SCHEMA_VERSION: Final[int] = 1
_CACHE_FILE_MODE: Final[int] = 0o600
_PROGRESS_INTERVAL: Final[int] = 25
_ERROR_MAX_CHARS: Final[int] = 300
_CONTROL_CHAR_RE: Final[regex.Pattern[str]] = regex.compile(r"\p{Cc}")

_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS pdf_texts (
        url_key TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        extractor_version INTEGER NOT NULL,
        upload_name TEXT,
        byte_count INTEGER,
        page_texts TEXT NOT NULL,
        error TEXT,
        extracted_at TEXT NOT NULL
    )
    """,
)


class PdfEmbed(BaseModel):
    """One PDF embed found in a block.

    Attributes:
        host_uid: The uid of the block whose string embeds the PDF.
        url: The PDF's Firebase Storage URL, exactly as the block carries it.
    """

    model_config = ConfigDict(frozen=True)

    host_uid: str
    url: str


class PdfStatus(enum.StrEnum):
    """The outcome of one PDF's extraction."""

    OK = "ok"
    """At least one page carries a text layer."""

    NO_TEXT = "no-text"
    """The PDF parsed, but no page carries a text layer (a scan; OCR territory)."""

    PASSWORD = "password"
    """The PDF is password-protected.  Terminal: an upload never changes."""

    PARSE_FAILED = "parse-failed"
    """PDFium could not read the file."""

    FETCH_FAILED = "fetch-failed"
    """The Local API could not serve the file (e.g. an upload that no longer exists)."""


class PdfText(BaseModel):
    """One cached extraction result.

    Attributes:
        status: The extraction's outcome.
        extractor_version: The :data:`EXTRACTOR_VERSION` that produced the entry.
        upload_name: The name the PDF was uploaded under, when the storage reveals it.
        byte_count: The decrypted file's size, when it was fetched.
        page_texts: One whitespace-collapsed text per page, in page order — empty for a
            failure, and ``""`` for a page without a text layer.
        error: The failure's description, for a failed status.
        extracted_at: ISO-8601 moment of the extraction.
    """

    model_config = ConfigDict(frozen=True)

    status: PdfStatus
    extractor_version: int
    upload_name: str | None = None
    byte_count: int | None = None
    page_texts: tuple[str, ...] = ()
    error: str | None = None
    extracted_at: str


class PdfFetchBudget(BaseModel):
    """How much one run may fetch through Roam Desktop; ``None`` leaves a bound off.

    A run checks the budget before each fetch and stops once any bound is reached, so the
    fetch that crosses the byte or time bound completes.  Bytes matter as well as counts:
    every PDF crosses Roam Desktop's main process as one base64 string, and a run moving
    ~1.3 GB through a long-lived Roam process crashed it (design plan, 5b).

    Attributes:
        max_pdfs: The most PDFs to fetch.
        max_bytes: Stop once this many decrypted bytes have been fetched.
        max_seconds: Stop once the run has spent this long fetching and extracting.
    """

    model_config = ConfigDict(frozen=True)

    max_pdfs: int | None = Field(default=None, ge=0)
    max_bytes: int | None = Field(default=None, ge=0)
    max_seconds: float | None = Field(default=None, ge=0)

    def exhausted(self, fetched: int, fetched_bytes: int, elapsed_seconds: float) -> bool:
        """Whether a run that has fetched this much, this long, must stop before the next fetch."""
        return (
            (self.max_pdfs is not None and fetched >= self.max_pdfs)
            or (self.max_bytes is not None and fetched_bytes >= self.max_bytes)
            or (self.max_seconds is not None and elapsed_seconds >= self.max_seconds)
        )


class ExtractionSummary(BaseModel):
    """What an extraction run did, by count.

    Attributes:
        distinct_count: Distinct PDF URLs among the embeds the run was given.
        current_count: PDFs whose cached entry needed no work this run.
        outcomes: This run's extractions by resulting status.
        fetched_bytes: Decrypted bytes this run fetched.
        pending_count: PDFs still needing extraction after the run (the budget held them back).
    """

    model_config = ConfigDict(frozen=True)

    distinct_count: int
    current_count: int
    outcomes: dict[PdfStatus, int]
    fetched_bytes: int = 0
    pending_count: int

    @property
    def extracted_count(self) -> int:
        """PDFs fetched and extracted this run, whatever their outcome."""
        return sum(self.outcomes.values())


@validate_call
def default_pdf_cache_path(graph_name: str) -> Path:
    """The conventional cache location: ``~/.cache/roam-semantic-search/<graph>.pdf-text.db``.

    Args:
        graph_name: Canonical name of the source graph.

    Returns:
        The PDF text cache's default database path.
    """
    return Path.home() / ".cache" / "roam-semantic-search" / f"{graph_name}.pdf-text.db"


@validate_call
def pdf_embeds(rows: Sequence[dict[str, object]]) -> list[PdfEmbed]:
    """Every PDF embed in the graph's blocks, in row order (a block may embed several).

    Args:
        rows: Pull-block rows, one per entity, as returned by the full-graph fetch.

    Returns:
        One :class:`PdfEmbed` per ``{{pdf: <url>}}`` / ``{{[[pdf]]: <url>}}`` occurrence
        whose URL is a Firebase Storage URL.
    """
    embeds: Final[list[PdfEmbed]] = []
    for row in rows:
        string: object = row.get("string")
        uid: object = row.get("uid")
        if not isinstance(string, str) or not isinstance(uid, str):
            continue
        embeds.extend(PdfEmbed(host_uid=uid, url=match.group("url")) for match in PDF_EMBED_RE.finditer(string))
    return embeds


@validate_call
def url_key(url: str) -> str:
    """The cache key of a PDF URL: its SHA-256 hex digest (the URL itself is never stored).

    Args:
        url: The PDF's storage URL, exactly as the block carries it.

    Returns:
        The URL's SHA-256 hex digest.
    """
    return hashlib.sha256(url.encode()).hexdigest()


def _clean_page_text(raw_text: str) -> str:
    """A page's text with control characters dropped and whitespace runs collapsed."""
    return " ".join(_CONTROL_CHAR_RE.sub(" ", raw_text).split())


@validate_call
def extracted_text(contents: bytes) -> tuple[PdfStatus, tuple[str, ...], str | None]:
    """Extract a PDF's text layer, page by page.

    Args:
        contents: The PDF's (decrypted) bytes.

    Returns:
        The status, one cleaned text per page (``""`` for a page with fewer than
        :data:`NO_TEXT_PAGE_MIN_CHARS` non-whitespace chars), and an error description
        for a failed status.
    """
    try:
        document: Final[pypdfium2.PdfDocument] = pypdfium2.PdfDocument(contents)
    except pypdfium2.PdfiumError as error:
        status: PdfStatus = (
            PdfStatus.PASSWORD if error.err_code == pdfium_raw.FPDF_ERR_PASSWORD else PdfStatus.PARSE_FAILED
        )
        return status, (), str(error)[:_ERROR_MAX_CHARS]
    page_texts: Final[list[str]] = []
    try:
        for page_index in range(len(document)):
            page: pypdfium2.PdfPage = document[page_index]
            textpage: pypdfium2.PdfTextPage = page.get_textpage()
            text: str = _clean_page_text(textpage.get_text_bounded())
            textpage.close()
            page.close()
            page_texts.append(text if len(text.replace(" ", "")) >= NO_TEXT_PAGE_MIN_CHARS else "")
    except pypdfium2.PdfiumError as error:
        return PdfStatus.PARSE_FAILED, (), str(error)[:_ERROR_MAX_CHARS]
    finally:
        document.close()
    return (PdfStatus.OK if any(page_texts) else PdfStatus.NO_TEXT), tuple(page_texts), None


def _connected_cache(cache_path: Path) -> sqlite3.Connection:
    """Open (creating if absent) the owner-only cache database, schema ensured."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.touch(mode=_CACHE_FILE_MODE, exist_ok=True)
    cache_path.chmod(_CACHE_FILE_MODE)
    connection: Final[sqlite3.Connection] = sqlite3.connect(cache_path)
    with connection:
        for statement in _SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(_CACHE_SCHEMA_VERSION),)
        )
    stored_version: Final[str] = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
    if stored_version != str(_CACHE_SCHEMA_VERSION):
        connection.close()
        raise ValueError(
            f"PDF text cache {cache_path} is schema v{stored_version}, not v{_CACHE_SCHEMA_VERSION}"
            " — delete it to rebuild"
        )
    return connection


def _read_entries(connection: sqlite3.Connection) -> dict[str, PdfText]:
    """Every cached entry by URL key."""
    rows: Final[list[tuple[str, str, int, str | None, int | None, str, str | None, str]]] = list(
        connection.execute(
            "SELECT url_key, status, extractor_version, upload_name, byte_count, page_texts, error, extracted_at"
            " FROM pdf_texts"
        )
    )
    return {
        key: PdfText(
            status=PdfStatus(status),
            extractor_version=extractor_version,
            upload_name=upload_name,
            byte_count=byte_count,
            page_texts=tuple(json.loads(page_texts)),
            error=error,
            extracted_at=extracted_at,
        )
        for key, status, extractor_version, upload_name, byte_count, page_texts, error, extracted_at in rows
    }


def _write_entry(connection: sqlite3.Connection, key: str, entry: PdfText) -> None:
    """Insert or replace one cached entry, committed at once so an interrupted run keeps it."""
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO pdf_texts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                entry.status.value,
                entry.extractor_version,
                entry.upload_name,
                entry.byte_count,
                json.dumps(list(entry.page_texts)),
                entry.error,
                entry.extracted_at,
            ),
        )


@validate_call
def cached_pdf_texts(cache_path: Path) -> dict[str, PdfText]:
    """Every cached extraction result, keyed by :func:`url_key`.

    Args:
        cache_path: The PDF text cache's database file.

    Returns:
        The cached entries by URL key; empty when the cache does not exist yet.
    """
    if not cache_path.exists():
        return {}
    connection: Final[sqlite3.Connection] = _connected_cache(cache_path)
    try:
        return _read_entries(connection)
    finally:
        connection.close()


def _needs_extraction(entry: PdfText | None, retry_failed: bool) -> bool:
    """Whether a URL's cached entry (or its absence) calls for a fetch this run.

    An upload never changes, so only a missing entry, a newer extractor, or an explicit
    retry of a failure brings a URL back; a password-protected PDF never comes back.
    """
    if entry is None:
        return True
    match entry.status:
        case PdfStatus.PASSWORD:
            return False
        case PdfStatus.FETCH_FAILED:
            return retry_failed
        case PdfStatus.PARSE_FAILED:
            return retry_failed or entry.extractor_version < EXTRACTOR_VERSION
        case PdfStatus.OK | PdfStatus.NO_TEXT:
            return entry.extractor_version < EXTRACTOR_VERSION


def _now() -> str:
    """The current moment, ISO-8601 to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetched_entry(source: GraphSource, url: str) -> PdfText:
    """Fetch one PDF into memory, extract it, and return its cache entry (bytes discarded).

    A Local API error answering for this file (an HTTP error status, or a malformed
    response) yields a :attr:`PdfStatus.FETCH_FAILED` entry; anything else — the Local API
    unreachable, say — propagates.
    """
    try:
        asset: Final[RoamAsset] = source.fetch_file(HttpUrl(url))
    except (requests.exceptions.HTTPError, ValidationError) as error:
        return PdfText(
            status=PdfStatus.FETCH_FAILED,
            extractor_version=EXTRACTOR_VERSION,
            error=f"{type(error).__name__}: {error}"[:_ERROR_MAX_CHARS],
            extracted_at=_now(),
        )
    status, page_texts, error_text = extracted_text(asset.contents)
    return PdfText(
        status=status,
        extractor_version=EXTRACTOR_VERSION,
        upload_name=asset.original_file_name,
        byte_count=len(asset.contents),
        page_texts=page_texts,
        error=error_text,
        extracted_at=_now(),
    )


@validate_call
def extract_pdf_texts(
    cache_path: Path,
    embeds: Sequence[PdfEmbed],
    source: GraphSource,
    retry_failed: bool = False,
    budget: PdfFetchBudget | None = None,
) -> ExtractionSummary:
    """Fetch and extract every embedded PDF the cache lacks, caching each result as it lands.

    Each distinct URL is visited once, in embed order.  A URL is fetched when the cache has
    no entry for it, when its entry came from an older extractor, or — with *retry_failed* —
    when its fetch or parse failed; a password-protected PDF is never refetched.  Every
    result is committed as soon as it lands, so an interrupted run keeps its progress, and
    a *budget* stops the run early, leaving the rest pending for the next one.

    A run of :data:`CONSECUTIVE_FETCH_FAILURE_LIMIT` failed fetches in a row means the graph
    itself is failing (closed, or locked mid-run), not its files: the run aborts *without*
    recording that run of failures, so they are retried next time rather than cached.

    Args:
        cache_path: The PDF text cache's database file; created owner-only if absent.
        embeds: The graph's PDF embeds (:func:`pdf_embeds`).
        source: The graph source the PDFs are fetched from.
        retry_failed: Whether to refetch PDFs whose fetch or parse failed before.
        budget: How much the run may fetch; ``None`` fetches everything that needs it.

    Returns:
        The run's summary.

    Raises:
        RuntimeError: If :data:`CONSECUTIVE_FETCH_FAILURE_LIMIT` fetches fail in a row.
    """
    urls: Final[list[str]] = list(dict.fromkeys(embed.url for embed in embeds))
    bounds: Final[PdfFetchBudget] = budget if budget is not None else PdfFetchBudget()
    connection: Final[sqlite3.Connection] = _connected_cache(cache_path)
    started: Final[float] = time.monotonic()
    fetched: int = 0
    fetched_bytes: int = 0
    try:
        cached: Final[dict[str, PdfText]] = _read_entries(connection)
        due: Final[list[str]] = [url for url in urls if _needs_extraction(cached.get(url_key(url)), retry_failed)]
        logger.info("%d distinct PDFs; %d need extraction", len(urls), len(due))
        outcomes: Final[Counter[PdfStatus]] = Counter()
        failure_run: Final[list[tuple[str, PdfText]]] = []
        for url in due:
            if bounds.exhausted(fetched, fetched_bytes, time.monotonic() - started):
                logger.info("fetch budget reached after %d PDFs (%d bytes)", fetched, fetched_bytes)
                break
            entry: PdfText = _fetched_entry(source, url)
            fetched += 1
            fetched_bytes += entry.byte_count or 0
            if entry.status is PdfStatus.FETCH_FAILED:
                failure_run.append((url, entry))
                if len(failure_run) >= CONSECUTIVE_FETCH_FAILURE_LIMIT:
                    raise RuntimeError(
                        f"{len(failure_run)} PDF fetches failed in a row (last: {entry.error}) — the graph is"
                        " likely closed or locked; nothing from that run was cached"
                    )
                continue
            for failed_url, failed_entry in failure_run:
                _write_entry(connection, url_key(failed_url), failed_entry)
                outcomes[failed_entry.status] += 1
            failure_run.clear()
            _write_entry(connection, url_key(url), entry)
            outcomes[entry.status] += 1
            if fetched % _PROGRESS_INTERVAL == 0:
                logger.info("extracted %d/%d PDFs", fetched, len(due))
        for failed_url, failed_entry in failure_run:
            _write_entry(connection, url_key(failed_url), failed_entry)
            outcomes[failed_entry.status] += 1
    finally:
        connection.close()
    return ExtractionSummary(
        distinct_count=len(urls),
        current_count=len(urls) - len(due),
        outcomes=dict(outcomes),
        fetched_bytes=fetched_bytes,
        pending_count=len(due) - fetched,
    )
