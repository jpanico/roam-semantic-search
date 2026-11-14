"""The on-disk index store: one SQLite file holding records, vectors, and keyword index.

The store is a single SQLite database: a ``records`` table (one row per index record,
its embedding as a float32 blob), an FTS5 ``records_fts`` mirror for keyword retrieval,
and a ``meta`` key/value table recording the embedding model, its dimension, and build
provenance.  Vector KNN is served brute-force from the blobs (the index is ~10k vectors;
a matrix product answers in milliseconds), so no SQLite extension is required.

Public symbols:

- :data:`SCHEMA_VERSION` — the store layout's version, stamped into the meta.
- :data:`BM25_WEIGHT_TEXT` / :data:`BM25_WEIGHT_BREADCRUMB` / :data:`BM25_WEIGHT_CONCEPTS` /
  :data:`BM25_WEIGHT_TAGS` / :data:`BM25_WEIGHT_DESCENDANT_TEXT` — the keyword ranking's
  per-column BM25 weights.
- :class:`StoreMeta` — the store's provenance and embedding-model facts.
- :class:`StoredRecord` — one record read back from the store.
- :func:`default_db_path` — the conventional store location for a graph.
- :func:`write_store` — full-rebuild write of records + embeddings + meta.
- :func:`upgrade_store` — migrate a schema-v2 store to v3 in place, embeddings kept.
- :func:`upsert_records` — insert-or-replace records + embeddings in place.
- :func:`delete_records` — remove records by record id.
- :func:`stamp_refresh` — record a refresh moment and record count in the meta.
- :func:`read_meta` — the store's :class:`StoreMeta`.
- :func:`stored_hashes` — every record's content hash by record id.
- :func:`load_embedding_matrix` — every record's embedding as one float32 matrix.
- :func:`keyword_ranked_ids` — BM25-ranked record ids for a query's terms.
- :func:`records_by_id` — read back records by record id.

Records are keyed by :attr:`~roam_semantic_search.normalize.IndexRecord.record_id`, not by
Roam uid: every chunk of a PDF a block embeds carries the block's uid (schema v3).
"""

import json
import logging
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import regex
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.normalize import IndexRecord, RecordKind

logger = logging.getLogger(__name__)

_WORD_RE: Final[regex.Pattern[str]] = regex.compile(r"\w+")

SCHEMA_VERSION: Final[int] = 3
"""The store layout's version, stamped into the meta.

A v2 store upgrades in place (:func:`upgrade_store`); any other mismatch needs a full rebuild.
"""

_UPGRADABLE_SCHEMA_VERSION: Final[int] = 2
_BUSY_TIMEOUT_SECONDS: Final[float] = 30.0
_STORE_FILE_MODE: Final[int] = 0o600
"""A store holds the graph's decrypted text, so it is owner-only."""

BM25_WEIGHT_TEXT: Final[float] = 1.0
"""Keyword-ranking weight of a record's own text (the base word weight)."""

BM25_WEIGHT_BREADCRUMB: Final[float] = 1.0
"""Keyword-ranking weight of a record's context path."""

BM25_WEIGHT_CONCEPTS: Final[float] = 4.0
"""Keyword-ranking weight of a record's referenced page names (the highest tier)."""

BM25_WEIGHT_TAGS: Final[float] = 2.0
"""Keyword-ranking weight of a record's ``tags::`` classification values (the middle tier)."""

BM25_WEIGHT_DESCENDANT_TEXT: Final[float] = 1.0
"""Keyword-ranking weight of a record's folded descendant text (the base word weight)."""

_RECORDS_COLUMNS: Final[str] = """(
        record_id TEXT PRIMARY KEY,
        uid TEXT NOT NULL,
        kind TEXT NOT NULL,
        page_title TEXT NOT NULL,
        breadcrumb TEXT NOT NULL,
        text TEXT NOT NULL,
        concepts TEXT NOT NULL,
        tags TEXT NOT NULL,
        descendant_text TEXT NOT NULL,
        embed_input TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        edited_at INTEGER NOT NULL,
        pdf_name TEXT,
        pdf_page INTEGER,
        embedding BLOB NOT NULL
    )"""
_FTS_COLUMNS: Final[str] = "fts5(text, breadcrumb, concepts, tags, descendant_text, record_id UNINDEXED)"

_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    f"CREATE TABLE records {_RECORDS_COLUMNS}",
    f"CREATE VIRTUAL TABLE records_fts USING {_FTS_COLUMNS}",
)

_V2_TO_V3_STATEMENTS: Final[tuple[str, ...]] = (
    f"CREATE TABLE records_v3 {_RECORDS_COLUMNS}",
    "INSERT INTO records_v3 (record_id, uid, kind, page_title, breadcrumb, text, concepts, tags,"
    " descendant_text, embed_input, content_hash, edited_at, pdf_name, pdf_page, embedding)"
    " SELECT uid, uid, CASE WHEN is_page THEN 'page' ELSE 'block' END, page_title, breadcrumb, text, concepts,"
    " tags, descendant_text, embed_input, content_hash, edited_at, NULL, NULL, embedding FROM records",
    "DROP TABLE records",
    "ALTER TABLE records_v3 RENAME TO records",
    f"CREATE VIRTUAL TABLE records_fts_v3 USING {_FTS_COLUMNS}",
    "INSERT INTO records_fts_v3 (text, breadcrumb, concepts, tags, descendant_text, record_id)"
    " SELECT text, breadcrumb, concepts, tags, descendant_text, uid FROM records_fts",
    "DROP TABLE records_fts",
    "ALTER TABLE records_fts_v3 RENAME TO records_fts",
    "UPDATE meta SET value = '3' WHERE key = 'schema_version'",
)


class StoreMeta(BaseModel):
    """The store's provenance and embedding-model facts.

    Attributes:
        graph_name: Name of the source graph.
        embed_model: Embedding model the vectors were produced with.
        dimension: The embedding vectors' dimension.
        built_at: ISO-8601 moment the store was fully built.
        record_count: Number of records in the store.
        refreshed_at: ISO-8601 moment of the latest incremental refresh, when one has run.
        schema_version: The store layout's version; ``None`` for a store predating versioning.
    """

    model_config = ConfigDict(frozen=True)

    graph_name: str
    embed_model: str
    dimension: int
    built_at: str
    record_count: int
    refreshed_at: str | None = None
    schema_version: int | None = None


class StoredRecord(BaseModel):
    """One index record read back from the store (embedding excluded).

    Attributes:
        record_id: The record's unique identity in the store.
        uid: The Roam uid it links to — for a PDF chunk, the block that embeds the PDF.
        kind: What the record represents: a page, a block, or a PDF chunk.
        page_title: Title of the page the record belongs to.
        breadcrumb: The record's context path.
        text: The record's plain text.
        pdf_name: For a PDF chunk, the name the PDF was uploaded under, when known.
        pdf_page: For a PDF chunk, the 1-based page it comes from.
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    uid: str
    kind: RecordKind
    page_title: str
    breadcrumb: str
    text: str
    pdf_name: str | None = None
    pdf_page: int | None = None

    @property
    def is_page(self) -> bool:
        """Whether the record represents a page."""
        return self.kind is RecordKind.PAGE


@validate_call
def default_db_path(graph_name: str) -> Path:
    """The conventional store location for a graph: ``~/.cache/roam-semantic-search/<graph>.db``.

    Args:
        graph_name: Name of the source graph.

    Returns:
        The store's default database path.
    """
    return Path.home() / ".cache" / "roam-semantic-search" / f"{graph_name}.db"


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def write_store(
    db_path: Path,
    records: Sequence[IndexRecord],
    embeddings: NDArray[np.float32],
    meta: StoreMeta,
) -> None:
    """Write a store from scratch: full rebuild, replacing any existing database file.

    Args:
        db_path: The database file to (re)create; parent directories are created as needed.
        records: The index records, in embedding row order.
        embeddings: One float32 embedding row per record.
        meta: The store's provenance and embedding-model facts.

    Raises:
        ValueError: If the record and embedding counts disagree.
    """
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"{len(records)} records but {embeddings.shape[0]} embeddings")
    stamped_meta: Final[StoreMeta] = (
        meta if meta.schema_version is not None else meta.model_copy(update={"schema_version": SCHEMA_VERSION})
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    db_path.touch(mode=_STORE_FILE_MODE)
    db_path.chmod(_STORE_FILE_MODE)
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [(key, str(value)) for key, value in stamped_meta.model_dump().items() if value is not None],
            )
            connection.executemany(_INSERT_RECORD, _record_rows(records, embeddings))
            connection.executemany(_INSERT_FTS, _fts_rows(records))
    finally:
        connection.close()


_INSERT_RECORD: Final[str] = "INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
_INSERT_FTS: Final[str] = (
    "INSERT INTO records_fts (text, breadcrumb, concepts, tags, descendant_text, record_id) VALUES (?, ?, ?, ?, ?, ?)"
)


@validate_call
def upgrade_store(db_path: Path) -> bool:
    """Migrate a schema-v2 store to v3 in place, keeping every embedding.

    v3 keys records by record id rather than Roam uid, records each record's kind, and gives
    PDF chunk records their page and upload name.  A v2 record is a page or block record, whose
    record id is its uid, so the migration copies rows rather than re-embedding: it needs
    neither Roam Desktop nor the embedding server, and it is a no-op on any store that is not
    v2.  It runs in one immediate transaction, so a concurrent caller either waits and then
    finds the store already upgraded, or sees it untouched.

    Whatever the version, it also makes the file owner-only (``0600``): a store holds the
    graph's decrypted text — attached PDFs' included — so a store an older version created
    world-readable is tightened the next time it is refreshed or searched.

    Args:
        db_path: The store's database file.

    Returns:
        Whether the store was migrated (``False`` when it was not v2).
    """
    db_path.chmod(_STORE_FILE_MODE)
    connection: Final[sqlite3.Connection] = sqlite3.connect(
        db_path, isolation_level=None, timeout=_BUSY_TIMEOUT_SECONDS
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        row: Final[tuple[str] | None] = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or row[0] != str(_UPGRADABLE_SCHEMA_VERSION):
            connection.execute("ROLLBACK")
            return False
        try:
            for statement in _V2_TO_V3_STATEMENTS:
                connection.execute(statement)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("VACUUM")
    finally:
        connection.close()
    logger.info("upgraded %s from schema v%d to v%d", db_path, _UPGRADABLE_SCHEMA_VERSION, SCHEMA_VERSION)
    return True


def _record_rows(
    records: Sequence[IndexRecord], embeddings: NDArray[np.float32]
) -> list[tuple[str, str, str, str, str, str, str, str, str, str, str, int, str | None, int | None, bytes]]:
    """The ``records`` table's row tuples for *records*, embeddings row-aligned."""
    return [
        (
            record.record_id,
            record.uid,
            record.kind.value,
            record.page_title,
            record.breadcrumb,
            record.text,
            json.dumps(list(record.concepts)),
            json.dumps(list(record.tags)),
            record.descendant_text,
            record.embed_input,
            record.content_hash,
            record.edited_at,
            record.pdf.name if record.pdf is not None else None,
            record.pdf.page if record.pdf is not None else None,
            embeddings[row_index].tobytes(),
        )
        for row_index, record in enumerate(records)
    ]


def _fts_rows(records: Sequence[IndexRecord]) -> list[tuple[str, str, str, str, str, str]]:
    """The FTS mirror's row tuples for *records* (list fields joined as plain text)."""
    return [
        (
            record.text,
            record.breadcrumb,
            " · ".join(record.concepts),
            " · ".join(record.tags),
            record.descendant_text,
            record.record_id,
        )
        for record in records
    ]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def upsert_records(
    db_path: Path,
    records: Sequence[IndexRecord],
    embeddings: NDArray[np.float32],
) -> None:
    """Insert-or-replace records and their embeddings in an existing store.

    Each record replaces any stored row sharing its record id, in both the ``records``
    table and the FTS mirror.

    Args:
        db_path: The store's database file.
        records: The records to upsert, in embedding row order.
        embeddings: One float32 embedding row per record.

    Raises:
        ValueError: If the record and embedding counts disagree.
    """
    if len(records) != embeddings.shape[0]:
        raise ValueError(f"{len(records)} records but {embeddings.shape[0]} embeddings")
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            connection.executemany(_INSERT_RECORD, _record_rows(records, embeddings))
            connection.executemany(
                "DELETE FROM records_fts WHERE record_id = ?", [(record.record_id,) for record in records]
            )
            connection.executemany(_INSERT_FTS, _fts_rows(records))
    finally:
        connection.close()


@validate_call
def delete_records(db_path: Path, record_ids: Sequence[str]) -> None:
    """Remove records by record id from the ``records`` table and the FTS mirror.

    Args:
        db_path: The store's database file.
        record_ids: The record ids to remove; absent ids are ignored.
    """
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            connection.executemany("DELETE FROM records WHERE record_id = ?", [(rid,) for rid in record_ids])
            connection.executemany("DELETE FROM records_fts WHERE record_id = ?", [(rid,) for rid in record_ids])
    finally:
        connection.close()


@validate_call
def stamp_refresh(db_path: Path, record_count: int) -> None:
    """Record the current moment as the store's latest refresh, with its record count.

    Args:
        db_path: The store's database file.
        record_count: The store's record count after the refresh.
    """
    refreshed_at: Final[str] = datetime.now(UTC).isoformat(timespec="seconds")
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("refreshed_at", refreshed_at), ("record_count", str(record_count))],
            )
    finally:
        connection.close()


@validate_call
def stored_hashes(db_path: Path) -> dict[str, str]:
    """Every stored record's content hash by record id.

    Args:
        db_path: The store's database file.

    Returns:
        The content hash keyed by record id, for the whole store.
    """
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, str]]] = list(connection.execute("SELECT record_id, content_hash FROM records"))
    finally:
        connection.close()
    return dict(rows)


@validate_call
def read_meta(db_path: Path) -> StoreMeta:
    """Read the store's :class:`StoreMeta`.

    Args:
        db_path: The store's database file.

    Returns:
        The store's provenance and embedding-model facts.
    """
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        pairs: Final[list[tuple[str, str]]] = list(connection.execute("SELECT key, value FROM meta"))
    finally:
        connection.close()
    return StoreMeta.model_validate(dict(pairs))


@validate_call
def load_embedding_matrix(db_path: Path) -> tuple[list[str], NDArray[np.float32]]:
    """Load every record's embedding as one matrix, with record ids in row order.

    Args:
        db_path: The store's database file.

    Returns:
        The record ids and the ``(len(record_ids), dimension)`` float32 embedding matrix.
    """
    dimension: Final[int] = read_meta(db_path).dimension
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, bytes]]] = list(connection.execute("SELECT record_id, embedding FROM records"))
    finally:
        connection.close()
    record_ids: Final[list[str]] = [record_id for record_id, _ in rows]
    if not rows:
        return record_ids, np.empty((0, dimension), dtype=np.float32)
    matrix: Final[NDArray[np.float32]] = np.frombuffer(b"".join(blob for _, blob in rows), dtype=np.float32).reshape(
        len(rows), dimension
    )
    return record_ids, matrix


@validate_call
def keyword_ranked_ids(db_path: Path, query_text: str, limit: int) -> list[str]:
    """BM25-ranked record ids for a query's terms, best first, per-column weighted.

    The query's word tokens are OR-joined as quoted FTS5 terms, so natural-language
    queries need no FTS syntax and cannot trip over it.  Ranking weighs where a term
    matches: a record's referenced page names (:data:`BM25_WEIGHT_CONCEPTS`) over its
    ``tags::`` values (:data:`BM25_WEIGHT_TAGS`) over its own, breadcrumb, and
    descendant text (the base weights).

    Args:
        db_path: The store's database file.
        query_text: The natural-language query.
        limit: Maximum record ids returned.

    Returns:
        The matching record ids, best-ranked first (empty when the query has no word tokens).
    """
    tokens: Final[list[str]] = _WORD_RE.findall(query_text)
    if not tokens:
        return []
    match_expression: Final[str] = " OR ".join(f'"{token}"' for token in tokens)
    ranking_expression: Final[str] = (
        f"bm25(records_fts, {BM25_WEIGHT_TEXT}, {BM25_WEIGHT_BREADCRUMB},"
        f" {BM25_WEIGHT_CONCEPTS}, {BM25_WEIGHT_TAGS}, {BM25_WEIGHT_DESCENDANT_TEXT})"
    )
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str]]] = list(
            connection.execute(
                f"SELECT record_id FROM records_fts WHERE records_fts MATCH ? ORDER BY {ranking_expression} LIMIT ?",
                (match_expression, limit),
            )
        )
    finally:
        connection.close()
    return [record_id for (record_id,) in rows]


@validate_call
def records_by_id(db_path: Path, record_ids: Sequence[str]) -> dict[str, StoredRecord]:
    """Read back records by record id.

    Args:
        db_path: The store's database file.
        record_ids: The record ids to read.

    Returns:
        The found records keyed by record id (missing ids simply absent).
    """
    if not record_ids:
        return {}
    placeholders: Final[str] = ",".join("?" for _ in record_ids)
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, str, str, str, str, str, str | None, int | None]]] = list(
            connection.execute(
                "SELECT record_id, uid, kind, page_title, breadcrumb, text, pdf_name, pdf_page"
                f" FROM records WHERE record_id IN ({placeholders})",
                list(record_ids),
            )
        )
    finally:
        connection.close()
    return {
        record_id: StoredRecord(
            record_id=record_id,
            uid=uid,
            kind=RecordKind(kind),
            page_title=page_title,
            breadcrumb=breadcrumb,
            text=text,
            pdf_name=pdf_name,
            pdf_page=pdf_page,
        )
        for record_id, uid, kind, page_title, breadcrumb, text, pdf_name, pdf_page in rows
    }
