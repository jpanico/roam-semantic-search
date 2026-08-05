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
- :func:`upsert_records` — insert-or-replace records + embeddings in place.
- :func:`delete_records` — remove records by uid.
- :func:`stamp_refresh` — record a refresh moment and record count in the meta.
- :func:`read_meta` — the store's :class:`StoreMeta`.
- :func:`stored_hashes` — every record's content hash by uid.
- :func:`load_embedding_matrix` — every record's embedding as one float32 matrix.
- :func:`keyword_ranked_uids` — BM25-ranked uids for a query's terms.
- :func:`records_by_uid` — read back records by uid.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import regex
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.normalize import IndexRecord

_WORD_RE: Final[regex.Pattern[str]] = regex.compile(r"\w+")

SCHEMA_VERSION: Final[int] = 2
"""The store layout's version, stamped into the meta; a mismatched store needs a full rebuild."""

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

_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE records (
        uid TEXT PRIMARY KEY,
        page_title TEXT NOT NULL,
        breadcrumb TEXT NOT NULL,
        text TEXT NOT NULL,
        concepts TEXT NOT NULL,
        tags TEXT NOT NULL,
        descendant_text TEXT NOT NULL,
        embed_input TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        edited_at INTEGER NOT NULL,
        is_page INTEGER NOT NULL,
        embedding BLOB NOT NULL
    )
    """,
    "CREATE VIRTUAL TABLE records_fts USING fts5(text, breadcrumb, concepts, tags, descendant_text, uid UNINDEXED)",
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
        uid: The record's stable identifier.
        page_title: Title of the page the record belongs to.
        breadcrumb: The record's context path.
        text: The record's plain text.
        is_page: Whether the record represents a page rather than a block.
    """

    model_config = ConfigDict(frozen=True)

    uid: str
    page_title: str
    breadcrumb: str
    text: str
    is_page: bool


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
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [(key, str(value)) for key, value in stamped_meta.model_dump().items() if value is not None],
            )
            connection.executemany(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _record_rows(records, embeddings),
            )
            connection.executemany(
                "INSERT INTO records_fts (text, breadcrumb, concepts, tags, descendant_text, uid)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                _fts_rows(records),
            )
    finally:
        connection.close()


def _record_rows(
    records: Sequence[IndexRecord], embeddings: NDArray[np.float32]
) -> list[tuple[str, str, str, str, str, str, str, str, str, int, int, bytes]]:
    """The ``records`` table's row tuples for *records*, embeddings row-aligned."""
    return [
        (
            record.uid,
            record.page_title,
            record.breadcrumb,
            record.text,
            json.dumps(list(record.concepts)),
            json.dumps(list(record.tags)),
            record.descendant_text,
            record.embed_input,
            record.content_hash,
            record.edited_at,
            int(record.is_page),
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
            record.uid,
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

    Each record replaces any stored row sharing its uid, in both the ``records``
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
            connection.executemany(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _record_rows(records, embeddings),
            )
            connection.executemany("DELETE FROM records_fts WHERE uid = ?", [(record.uid,) for record in records])
            connection.executemany(
                "INSERT INTO records_fts (text, breadcrumb, concepts, tags, descendant_text, uid)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                _fts_rows(records),
            )
    finally:
        connection.close()


@validate_call
def delete_records(db_path: Path, uids: Sequence[str]) -> None:
    """Remove records by uid from the ``records`` table and the FTS mirror.

    Args:
        db_path: The store's database file.
        uids: The uids to remove; absent uids are ignored.
    """
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        with connection:
            connection.executemany("DELETE FROM records WHERE uid = ?", [(uid,) for uid in uids])
            connection.executemany("DELETE FROM records_fts WHERE uid = ?", [(uid,) for uid in uids])
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
    """Every stored record's content hash by uid.

    Args:
        db_path: The store's database file.

    Returns:
        The content hash keyed by uid, for the whole store.
    """
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, str]]] = list(connection.execute("SELECT uid, content_hash FROM records"))
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
    """Load every record's embedding as one matrix, with uids in row order.

    Args:
        db_path: The store's database file.

    Returns:
        The uids and the ``(len(uids), dimension)`` float32 embedding matrix.
    """
    dimension: Final[int] = read_meta(db_path).dimension
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, bytes]]] = list(connection.execute("SELECT uid, embedding FROM records"))
    finally:
        connection.close()
    uids: Final[list[str]] = [uid for uid, _ in rows]
    if not rows:
        return uids, np.empty((0, dimension), dtype=np.float32)
    matrix: Final[NDArray[np.float32]] = np.frombuffer(b"".join(blob for _, blob in rows), dtype=np.float32).reshape(
        len(rows), dimension
    )
    return uids, matrix


@validate_call
def keyword_ranked_uids(db_path: Path, query_text: str, limit: int) -> list[str]:
    """BM25-ranked uids for a query's terms, best first, per-column weighted.

    The query's word tokens are OR-joined as quoted FTS5 terms, so natural-language
    queries need no FTS syntax and cannot trip over it.  Ranking weighs where a term
    matches: a record's referenced page names (:data:`BM25_WEIGHT_CONCEPTS`) over its
    ``tags::`` values (:data:`BM25_WEIGHT_TAGS`) over its own, breadcrumb, and
    descendant text (the base weights).

    Args:
        db_path: The store's database file.
        query_text: The natural-language query.
        limit: Maximum uids returned.

    Returns:
        The matching uids, best-ranked first (empty when the query has no word tokens).
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
                f"SELECT uid FROM records_fts WHERE records_fts MATCH ? ORDER BY {ranking_expression} LIMIT ?",
                (match_expression, limit),
            )
        )
    finally:
        connection.close()
    return [uid for (uid,) in rows]


@validate_call
def records_by_uid(db_path: Path, uids: Sequence[str]) -> dict[str, StoredRecord]:
    """Read back records by uid.

    Args:
        db_path: The store's database file.
        uids: The uids to read.

    Returns:
        The found records keyed by uid (missing uids simply absent).
    """
    if not uids:
        return {}
    placeholders: Final[str] = ",".join("?" for _ in uids)
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    try:
        rows: Final[list[tuple[str, str, str, str, int]]] = list(
            connection.execute(
                f"SELECT uid, page_title, breadcrumb, text, is_page FROM records WHERE uid IN ({placeholders})",
                list(uids),
            )
        )
    finally:
        connection.close()
    return {
        uid: StoredRecord(uid=uid, page_title=page_title, breadcrumb=breadcrumb, text=text, is_page=bool(is_page))
        for uid, page_title, breadcrumb, text, is_page in rows
    }
