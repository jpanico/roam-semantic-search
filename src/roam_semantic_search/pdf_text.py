"""Attachment text: find a graph's PDF and SVG attachments, extract their text, and cache it.

SVGs (Phase 6a) share the PDF pipeline end to end — discovery, the decrypting fetch, the
budget, and the cache — and differ only in extraction: an SVG's words are in its XML
(:func:`extracted_svg_text`), so it needs no OCR, and it is kept as a one-page document.
The module and its cache file keep their PDF-era names.

A PDF sits in a block as ``{{pdf: <url>}}`` or ``{{[[pdf]]: <url>}}``; the full-graph pull
carries only that markup, never the file.  This module reaches the content: each distinct URL
is fetched through the Local API's ``file.get`` action (which returns an encrypted graph's
asset decrypted), held in memory only, its text layer extracted page by page with pypdfium2,
and the result cached by URL.  A Firebase Storage URL names one immutable upload, so a cached
entry never goes stale; only a newer extractor (:data:`EXTRACTOR_VERSION`) re-extracts it.

A page without a text layer — a scan — can be read by on-device OCR
(:mod:`roam_semantic_search.ocr`).  OCR is opt-in per run, since it costs about a second a
page: a run with ``ocr=True`` fills such pages, and a cached PDF with blank pages that no
current OCR has read (:data:`~roam_semantic_search.ocr.OCR_VERSION`) is fetched again for it.

The cache is its own SQLite file beside the index store, because ``build`` deletes and
recreates the store and a rebuild must not refetch every PDF.  It holds decrypted text, so it
is kept owner-only (``0600``).  No PDF bytes are ever written to disk, and the cache keys
entries by a digest of the URL, never the URL itself (a Firebase URL embeds an access token).

Public symbols:

- :class:`Media` — what kind of attachment an embed or cache entry is: a PDF or an SVG.
- :class:`PdfEmbed` — one attachment: the hosting block's uid, its storage URL, its media.
- :class:`PdfStatus` — the outcome of one PDF's extraction.
- :class:`PdfText` — one cached extraction result.
- :class:`ExtractedText` — what extracting one PDF found: status, page texts, OCR'd pages.
- :class:`PdfFetchBudget` — how much one run may fetch: PDFs, bytes, seconds.
- :class:`ExtractionSummary` — what an extraction run did, by count.
- :data:`EXTRACTOR_VERSION` — the extraction logic's version; a bump re-extracts.
- :data:`NO_TEXT_PAGE_MIN_CHARS` — the non-whitespace chars a page needs to count as text.
- :data:`CONSECUTIVE_FETCH_FAILURE_LIMIT` — failed fetches in a row that abort a run.
- :func:`default_pdf_cache_path` — the conventional cache location for a graph.
- :func:`pdf_embeds` — pull-block rows → every PDF embed, in row order.
- :func:`svg_embeds` — pull-block rows → every SVG upload a block names, in row order.
- :func:`extracted_svg_text` — SVG bytes → status and text, read from the XML.
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
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import pypdfium2
import pypdfium2.raw as pdfium_raw
import regex
import requests
from guffin.roam.asset import RoamAsset
from guffin.roam.graph_source import GraphSource
from guffin.roam.markdown import FIREBASE_STORAGE_URL_RE, PDF_EMBED_RE
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, validate_call

from roam_semantic_search.ocr import OCR_VERSION, ocr_available, recognized_page_text

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION: Final[int] = 1
"""The extraction logic's version; entries extracted by an older one are re-extracted."""

NO_TEXT_PAGE_MIN_CHARS: Final[int] = 20
"""Non-whitespace text-layer chars a page needs to count as carrying text (fewer: a scan)."""

CONSECUTIVE_FETCH_FAILURE_LIMIT: Final[int] = 5
"""Failed fetches in a row that abort a run: the graph itself, not the files, is failing."""

_CACHE_SCHEMA_VERSION: Final[int] = 3
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
        extracted_at TEXT NOT NULL,
        ocr_version INTEGER NOT NULL DEFAULT 0,
        ocr_page_count INTEGER NOT NULL DEFAULT 0,
        media TEXT NOT NULL DEFAULT 'pdf'
    )
    """,
)

_V1_TO_V2_STATEMENTS: Final[tuple[str, ...]] = (
    "ALTER TABLE pdf_texts ADD COLUMN ocr_version INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE pdf_texts ADD COLUMN ocr_page_count INTEGER NOT NULL DEFAULT 0",
    "UPDATE meta SET value = '2' WHERE key = 'schema_version'",
)

_V2_TO_V3_STATEMENTS: Final[tuple[str, ...]] = (
    "ALTER TABLE pdf_texts ADD COLUMN media TEXT NOT NULL DEFAULT 'pdf'",
    "UPDATE meta SET value = '3' WHERE key = 'schema_version'",
)

_SVG_TEXT_TAGS: Final[frozenset[str]] = frozenset({"title", "desc", "text", "foreignObject"})
"""SVG elements whose whole text is read (``<text>`` with its ``<tspan>`` / ``<textPath>``)."""

_SVG_SKIPPED_TAGS: Final[frozenset[str]] = frozenset({"style", "script", "metadata"})
"""SVG elements whose text is never words a reader sees."""

_SVG_SUFFIXES: Final[tuple[str, ...]] = (".svg", ".svg.enc")


class Media(enum.StrEnum):
    """What kind of attachment an embed or cache entry is."""

    PDF = "pdf"
    SVG = "svg"


class PdfEmbed(BaseModel):
    """One attachment found in a block — a PDF embed, or (``media`` SVG) an SVG upload.

    Attributes:
        host_uid: The uid of the block whose string carries the attachment.
        url: The attachment's Firebase Storage URL, exactly as the block carries it.
        media: What kind of attachment it is.
    """

    model_config = ConfigDict(frozen=True)

    host_uid: str
    url: str
    media: Media = Media.PDF


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
        ocr_version: The :data:`~roam_semantic_search.ocr.OCR_VERSION` that read the PDF's
            blank pages, or ``0`` when no OCR has.
        ocr_page_count: How many of ``page_texts`` came from OCR.
        media: What kind of attachment the entry is (an SVG has one page).
    """

    model_config = ConfigDict(frozen=True)

    status: PdfStatus
    extractor_version: int
    upload_name: str | None = None
    byte_count: int | None = None
    page_texts: tuple[str, ...] = ()
    error: str | None = None
    extracted_at: str
    ocr_version: int = 0
    ocr_page_count: int = 0
    media: Media = Media.PDF


class ExtractedText(NamedTuple):
    """What extracting one PDF found.

    Attributes:
        status: The extraction's outcome.
        page_texts: One cleaned text per page; ``""`` for a page with no readable text.
        error: The failure's description, for a failed status.
        ocr_page_count: How many pages' text came from OCR.
    """

    status: PdfStatus
    page_texts: tuple[str, ...]
    error: str | None
    ocr_page_count: int = 0


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
        ocr_page_count: Pages this run read by OCR.
        pending_count: PDFs still needing extraction after the run (the budget held them back).
    """

    model_config = ConfigDict(frozen=True)

    distinct_count: int
    current_count: int
    outcomes: dict[PdfStatus, int]
    fetched_bytes: int = 0
    ocr_page_count: int = 0
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
def svg_embeds(rows: Sequence[dict[str, object]]) -> list[PdfEmbed]:
    """Every SVG upload the graph's blocks name, in row order — as an image embed or a bare URL.

    Args:
        rows: Pull-block rows, one per entity, as returned by the full-graph fetch.

    Returns:
        One :class:`PdfEmbed` (media SVG) per Firebase Storage URL whose object path ends
        ``.svg`` or ``.svg.enc``.
    """
    embeds: Final[list[PdfEmbed]] = []
    for row in rows:
        string: object = row.get("string")
        uid: object = row.get("uid")
        if not isinstance(string, str) or not isinstance(uid, str) or ".svg" not in string.lower():
            continue
        embeds.extend(
            PdfEmbed(host_uid=uid, url=match.group(0), media=Media.SVG)
            for match in FIREBASE_STORAGE_URL_RE.finditer(string)
            if match.group("object_path").lower().endswith(_SVG_SUFFIXES)
        )
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


def _readable(text: str) -> str:
    """*text* when it has at least :data:`NO_TEXT_PAGE_MIN_CHARS` non-whitespace chars, else ``""``."""
    return text if len(text.replace(" ", "")) >= NO_TEXT_PAGE_MIN_CHARS else ""


@validate_call
def extracted_text(contents: bytes, ocr: bool = False) -> ExtractedText:
    """Extract a PDF's text, page by page: its text layer, and with *ocr* its scanned pages.

    Args:
        contents: The PDF's (decrypted) bytes.
        ocr: Whether to OCR the pages without a usable text layer (on-device, when available).

    Returns:
        The status, one cleaned text per page (``""`` for a page with fewer than
        :data:`NO_TEXT_PAGE_MIN_CHARS` non-whitespace chars, after OCR when asked), an error
        description for a failed status, and how many pages OCR filled.
    """
    try:
        document: Final[pypdfium2.PdfDocument] = pypdfium2.PdfDocument(contents)
    except pypdfium2.PdfiumError as error:
        status: PdfStatus = (
            PdfStatus.PASSWORD if error.err_code == pdfium_raw.FPDF_ERR_PASSWORD else PdfStatus.PARSE_FAILED
        )
        return ExtractedText(status, (), str(error)[:_ERROR_MAX_CHARS])
    page_texts: Final[list[str]] = []
    ocr_page_count: int = 0
    try:
        for page_index in range(len(document)):
            page: pypdfium2.PdfPage = document[page_index]
            textpage: pypdfium2.PdfTextPage = page.get_textpage()
            text: str = _readable(_clean_page_text(textpage.get_text_bounded()))
            textpage.close()
            if not text and ocr:
                text = _readable(_clean_page_text(recognized_page_text(page)))
                ocr_page_count += 1 if text else 0
            page.close()
            page_texts.append(text)
    except pypdfium2.PdfiumError as error:
        return ExtractedText(PdfStatus.PARSE_FAILED, (), str(error)[:_ERROR_MAX_CHARS])
    finally:
        document.close()
    status_found: Final[PdfStatus] = PdfStatus.OK if any(page_texts) else PdfStatus.NO_TEXT
    return ExtractedText(status_found, tuple(page_texts), None, ocr_page_count)


def _svg_text_pieces(element: ElementTree.Element) -> list[str]:
    """The text an SVG element shows, in document order (skipped elements contribute nothing)."""
    tag: Final[str] = element.tag.rsplit("}", 1)[-1]
    if tag in _SVG_SKIPPED_TAGS:
        return []
    if tag in _SVG_TEXT_TAGS:
        return [" ".join(element.itertext())]
    return [piece for child in element for piece in _svg_text_pieces(child)]


@validate_call
def extracted_svg_text(contents: bytes) -> ExtractedText:
    """Read an SVG's text from its XML: titles, descriptions, text runs, and foreign objects.

    The words of a ``<text>`` element (with its ``<tspan>`` / ``<textPath>`` children),
    ``<title>``, ``<desc>``, and ``<foreignObject>`` (where Mermaid and draw.io put labels)
    are collected in document order; ``<style>``, ``<script>``, and ``<metadata>`` are
    skipped.  The standard-library parser resolves no external entities, and the expat it
    is built on guards against entity-expansion bombs.  Text drawn as outlines is not
    recoverable this way (that would take rendering and OCR).

    Args:
        contents: The SVG's (decrypted) bytes.

    Returns:
        One cleaned "page": OK when it has at least :data:`NO_TEXT_PAGE_MIN_CHARS`
        non-whitespace chars, else NO_TEXT; PARSE_FAILED for XML that does not parse.
    """
    try:
        root: Final[ElementTree.Element] = ElementTree.fromstring(contents)
    except ElementTree.ParseError as error:
        return ExtractedText(PdfStatus.PARSE_FAILED, (), str(error)[:_ERROR_MAX_CHARS])
    text: Final[str] = _readable(_clean_page_text("\n".join(_svg_text_pieces(root))))
    return ExtractedText(PdfStatus.OK if text else PdfStatus.NO_TEXT, (text,), None)


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
        if connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "1":
            for statement in _V1_TO_V2_STATEMENTS:  # v2 only adds the OCR columns; every entry is kept
                connection.execute(statement)
        if connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "2":
            for statement in _V2_TO_V3_STATEMENTS:  # v3 only adds the media column; every entry is a PDF
                connection.execute(statement)
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
    rows: Final[list[tuple[str, str, int, str | None, int | None, str, str | None, str, int, int, str]]] = list(
        connection.execute(
            "SELECT url_key, status, extractor_version, upload_name, byte_count, page_texts, error, extracted_at,"
            " ocr_version, ocr_page_count, media FROM pdf_texts"
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
            ocr_version=ocr_version,
            ocr_page_count=ocr_page_count,
            media=Media(media),
        )
        for (
            key,
            status,
            extractor_version,
            upload_name,
            byte_count,
            page_texts,
            error,
            extracted_at,
            ocr_version,
            ocr_page_count,
            media,
        ) in rows
    }


def _write_entry(connection: sqlite3.Connection, key: str, entry: PdfText) -> None:
    """Insert or replace one cached entry, committed at once so an interrupted run keeps it."""
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO pdf_texts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                entry.status.value,
                entry.extractor_version,
                entry.upload_name,
                entry.byte_count,
                json.dumps(list(entry.page_texts)),
                entry.error,
                entry.extracted_at,
                entry.ocr_version,
                entry.ocr_page_count,
                entry.media.value,
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


def _needs_ocr(entry: PdfText) -> bool:
    """Whether a cached PDF has blank pages that no current OCR has read (never an SVG)."""
    return entry.media is Media.PDF and "" in entry.page_texts and entry.ocr_version < OCR_VERSION


def _needs_extraction(entry: PdfText | None, retry_failed: bool, ocr: bool) -> bool:
    """Whether a URL's cached entry (or its absence) calls for a fetch this run.

    An upload never changes, so only a missing entry, a newer extractor, an explicit retry
    of a failure, or — in a run that OCRs — blank pages no current OCR has read bring a URL
    back; a password-protected PDF never comes back.
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
            return entry.extractor_version < EXTRACTOR_VERSION or (ocr and _needs_ocr(entry))


def _now() -> str:
    """The current moment, ISO-8601 to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetched_entry(source: GraphSource, url: str, ocr: bool, media: Media) -> PdfText:
    """Fetch one attachment into memory, extract it, and return its cache entry (bytes discarded).

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
            media=media,
        )
    extraction: Final[ExtractedText] = (
        extracted_svg_text(asset.contents) if media is Media.SVG else extracted_text(asset.contents, ocr=ocr)
    )
    return PdfText(
        status=extraction.status,
        extractor_version=EXTRACTOR_VERSION,
        upload_name=asset.original_file_name,
        byte_count=len(asset.contents),
        page_texts=extraction.page_texts,
        error=extraction.error,
        extracted_at=_now(),
        ocr_version=OCR_VERSION if ocr and media is Media.PDF else 0,
        ocr_page_count=extraction.ocr_page_count,
        media=media,
    )


@validate_call
def extract_pdf_texts(
    cache_path: Path,
    embeds: Sequence[PdfEmbed],
    source: GraphSource,
    retry_failed: bool = False,
    budget: PdfFetchBudget | None = None,
    ocr: bool = False,
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
        ocr: Whether to OCR pages without a text layer — and so to refetch cached PDFs whose
            blank pages no current OCR has read.  Ignored (with a warning) where on-device
            OCR is unavailable.

    Returns:
        The run's summary.

    Raises:
        RuntimeError: If :data:`CONSECUTIVE_FETCH_FAILURE_LIMIT` fetches fail in a row.
    """
    media_by_url: Final[dict[str, Media]] = {}
    for embed in embeds:
        media_by_url.setdefault(embed.url, embed.media)
    urls: Final[list[str]] = list(media_by_url)
    bounds: Final[PdfFetchBudget] = budget if budget is not None else PdfFetchBudget()
    run_ocr: Final[bool] = ocr and ocr_available()
    if ocr and not run_ocr:
        logger.warning("on-device OCR is unavailable here (macOS Vision via pyobjc); extracting text layers only")
    ocr_pages: int = 0
    connection: Final[sqlite3.Connection] = _connected_cache(cache_path)
    started: Final[float] = time.monotonic()
    fetched: int = 0
    fetched_bytes: int = 0
    try:
        cached: Final[dict[str, PdfText]] = _read_entries(connection)
        due: Final[list[str]] = [
            url for url in urls if _needs_extraction(cached.get(url_key(url)), retry_failed, run_ocr)
        ]
        logger.info("%d distinct attachments; %d need extraction", len(urls), len(due))
        outcomes: Final[Counter[PdfStatus]] = Counter()
        failure_run: Final[list[tuple[str, PdfText]]] = []
        for url in due:
            if bounds.exhausted(fetched, fetched_bytes, time.monotonic() - started):
                logger.info("fetch budget reached after %d PDFs (%d bytes)", fetched, fetched_bytes)
                break
            entry: PdfText = _fetched_entry(source, url, run_ocr, media_by_url[url])
            fetched += 1
            fetched_bytes += entry.byte_count or 0
            ocr_pages += entry.ocr_page_count
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
        ocr_page_count=ocr_pages,
        pending_count=len(due) - fetched,
    )
