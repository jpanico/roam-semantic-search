"""Tests for roam_semantic_search.store."""

import sqlite3
import stat
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from roam_semantic_search.normalize import IndexRecord, PdfLocator, RecordKind
from roam_semantic_search.store import (
    SCHEMA_VERSION,
    StoreMeta,
    delete_records,
    keyword_ranked_ids,
    load_embedding_matrix,
    read_meta,
    records_by_id,
    stamp_refresh,
    stored_hashes,
    upgrade_store,
    upsert_records,
    write_store,
)


def _record(
    uid: str,
    text: str,
    concepts: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    descendant_text: str = "",
) -> IndexRecord:
    return IndexRecord(
        uid=uid,
        page_title="P",
        breadcrumb=f"P › {text}",
        text=text,
        concepts=concepts,
        tags=tags,
        descendant_text=descendant_text,
        embed_input=f"P › {text}",
        content_hash="0" * 64,
        edited_at=1000,
        is_page=False,
    )


def _built_store(db_path: Path) -> tuple[list[IndexRecord], NDArray[np.float32]]:
    records: Final[list[IndexRecord]] = [
        _record("uid000001", "the hippo swims"),
        _record("uid000002", "a bird flies"),
    ]
    embeddings: Final[NDArray[np.float32]] = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    meta: Final[StoreMeta] = StoreMeta(
        graph_name="G", embed_model="m", dimension=3, built_at="2026-08-04T00:00:00+00:00", record_count=2
    )
    write_store(db_path, records, embeddings, meta)
    return records, embeddings


class TestStoreRoundTrip:
    def test_meta_round_trips(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        meta: Final[StoreMeta] = read_meta(db_path)
        assert meta.graph_name == "G"
        assert meta.dimension == 3
        assert meta.record_count == 2

    def test_embedding_matrix_round_trips(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _, embeddings = _built_store(db_path)
        uids, matrix = load_embedding_matrix(db_path)
        assert uids == ["uid000001", "uid000002"]
        assert np.array_equal(matrix, embeddings)

    def test_keyword_search_finds_term(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert keyword_ranked_ids(db_path, "hippo", 10) == ["uid000001"]

    def test_keyword_search_survives_fts_syntax(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert keyword_ranked_ids(db_path, 'hippo AND "NEAR(', 10) == ["uid000001"]

    def test_schema_version_stamped(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert read_meta(db_path).schema_version == SCHEMA_VERSION

    def test_records_read_back(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        records = records_by_id(db_path, ["uid000002", "missing00"])
        assert set(records) == {"uid000002"}
        assert records["uid000002"].text == "a bird flies"

    def test_rebuild_replaces_store(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        one: Final[list[IndexRecord]] = [_record("uid000009", "only one")]
        vectors: Final[NDArray[np.float32]] = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-08-04T01:00:00+00:00", record_count=1
        )
        write_store(db_path, one, vectors, meta)
        uids, _ = load_embedding_matrix(db_path)
        assert uids == ["uid000009"]


class TestWeightedKeywordRanking:
    def test_concept_match_outranks_tag_match_outranks_text_match(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        records: Final[list[IndexRecord]] = [
            _record("uid0text1", "the hippo swims here"),
            _record("uid00tag1", "a bird flies", tags=("hippo",)),
            _record("uid0conc1", "a whale sings", concepts=("hippo",)),
        ]
        vectors: Final[NDArray[np.float32]] = np.zeros((3, 3), dtype=np.float32)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-08-05T00:00:00+00:00", record_count=3
        )
        write_store(db_path, records, vectors, meta)
        assert keyword_ranked_ids(db_path, "hippo", 10) == ["uid0conc1", "uid00tag1", "uid0text1"]

    def test_descendant_text_matches_at_base_weight(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        records: Final[list[IndexRecord]] = [
            _record("uid0desc1", "a parent block", descendant_text="the hippo swims below"),
        ]
        vectors: Final[NDArray[np.float32]] = np.zeros((1, 3), dtype=np.float32)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-08-05T00:00:00+00:00", record_count=1
        )
        write_store(db_path, records, vectors, meta)
        assert keyword_ranked_ids(db_path, "hippo", 10) == ["uid0desc1"]


class TestStoreMutation:
    def test_upsert_replaces_and_adds(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        changed: Final[IndexRecord] = _record("uid000001", "the hippo now flies")
        added: Final[IndexRecord] = _record("uid000003", "a whale sings")
        vectors: Final[NDArray[np.float32]] = np.asarray([[0.0, 0.0, 1.0], [0.5, 0.5, 0.0]], dtype=np.float32)
        upsert_records(db_path, [changed, added], vectors)
        uids, matrix = load_embedding_matrix(db_path)
        assert set(uids) == {"uid000001", "uid000002", "uid000003"}
        assert np.array_equal(matrix[uids.index("uid000001")], vectors[0])
        assert keyword_ranked_ids(db_path, "whale", 10) == ["uid000003"]
        assert keyword_ranked_ids(db_path, "swims", 10) == []  # replaced text left the FTS mirror

    def test_delete_removes_record_and_fts_row(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        delete_records(db_path, ["uid000001", "missing00"])
        uids, _ = load_embedding_matrix(db_path)
        assert uids == ["uid000002"]
        assert keyword_ranked_ids(db_path, "hippo", 10) == []

    def test_stored_hashes_round_trip(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert stored_hashes(db_path) == {"uid000001": "0" * 64, "uid000002": "0" * 64}

    def test_stamp_refresh_updates_meta(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert read_meta(db_path).refreshed_at is None
        stamp_refresh(db_path, 5)
        meta: Final[StoreMeta] = read_meta(db_path)
        assert meta.refreshed_at is not None
        assert meta.record_count == 5


def _chunk(uid: str, page: int, text: str) -> IndexRecord:
    return _record(uid, text).model_copy(
        update={"pdf": PdfLocator(key="abcd1234", page=page, chunk=1, name="report.pdf"), "breadcrumb": "P › PDF"}
    )


class TestPdfChunkRecords:
    def test_chunks_share_their_host_uid_but_not_their_record_id(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        records: Final[list[IndexRecord]] = [
            _record("uid000001", "the host block"),
            _chunk("uid000001", 1, "the warranty covers the panels"),
            _chunk("uid000001", 2, "the invoice total"),
        ]
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-11-13T00:00:00+00:00", record_count=3
        )
        write_store(db_path, records, np.eye(3, dtype=np.float32), meta)
        record_ids, _ = load_embedding_matrix(db_path)
        assert record_ids == ["uid000001", "uid000001/pdf/abcd1234/1.1", "uid000001/pdf/abcd1234/2.1"]
        stored = records_by_id(db_path, ["uid000001/pdf/abcd1234/2.1"])["uid000001/pdf/abcd1234/2.1"]
        assert (stored.uid, stored.kind, stored.pdf_page, stored.pdf_name) == (
            "uid000001",
            RecordKind.PDF,
            2,
            "report.pdf",
        )
        assert not stored.is_page
        assert keyword_ranked_ids(db_path, "warranty", 10) == ["uid000001/pdf/abcd1234/1.1"]
        delete_records(db_path, ["uid000001/pdf/abcd1234/1.1"])
        assert load_embedding_matrix(db_path)[0] == ["uid000001", "uid000001/pdf/abcd1234/2.1"]
        assert keyword_ranked_ids(db_path, "warranty", 10) == []


_V2_SCHEMA: Final[tuple[str, ...]] = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE records (uid TEXT PRIMARY KEY, page_title TEXT NOT NULL, breadcrumb TEXT NOT NULL,"
    " text TEXT NOT NULL, concepts TEXT NOT NULL, tags TEXT NOT NULL, descendant_text TEXT NOT NULL,"
    " embed_input TEXT NOT NULL, content_hash TEXT NOT NULL, edited_at INTEGER NOT NULL,"
    " is_page INTEGER NOT NULL, embedding BLOB NOT NULL)",
    "CREATE VIRTUAL TABLE records_fts USING fts5(text, breadcrumb, concepts, tags, descendant_text, uid UNINDEXED)",
)


def _v2_store(db_path: Path) -> NDArray[np.float32]:
    """A store exactly as schema v2 wrote it: one page and one block record."""
    embeddings: Final[NDArray[np.float32]] = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    connection: Final[sqlite3.Connection] = sqlite3.connect(db_path)
    with connection:
        for statement in _V2_SCHEMA:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("graph_name", "G"),
                ("embed_model", "m"),
                ("dimension", "3"),
                ("built_at", "2026-08-05T00:00:00+00:00"),
                ("record_count", "2"),
                ("schema_version", "2"),
            ],
        )
        connection.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "page00001",
                    "Hippos",
                    "Hippos",
                    "Hippos",
                    "[]",
                    "[]",
                    "",
                    "Hippos",
                    "h1",
                    1000,
                    1,
                    embeddings[0].tobytes(),
                ),
                (
                    "uid000001",
                    "Hippos",
                    "Hippos",
                    "a bird flies",
                    '["Hippo"]',
                    "[]",
                    "",
                    "Hippos › a bird flies",
                    "h2",
                    1000,
                    0,
                    embeddings[1].tobytes(),
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO records_fts VALUES (?, ?, ?, ?, ?, ?)",
            [("Hippos", "Hippos", "", "", "", "page00001"), ("a bird flies", "Hippos", "Hippo", "", "", "uid000001")],
        )
    connection.close()
    return embeddings


class TestUpgradeStore:
    def test_v2_store_upgrades_in_place_keeping_embeddings(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        embeddings: Final[NDArray[np.float32]] = _v2_store(db_path)
        assert upgrade_store(db_path) is True
        assert read_meta(db_path).schema_version == SCHEMA_VERSION
        record_ids, matrix = load_embedding_matrix(db_path)
        assert record_ids == ["page00001", "uid000001"]
        assert np.array_equal(matrix, embeddings)
        stored = records_by_id(db_path, record_ids)
        assert stored["page00001"].kind is RecordKind.PAGE
        assert stored["uid000001"].kind is RecordKind.BLOCK
        assert stored["uid000001"].uid == "uid000001"
        assert stored_hashes(db_path) == {"page00001": "h1", "uid000001": "h2"}
        assert keyword_ranked_ids(db_path, "hippo", 10)[0] == "uid000001"  # the concept column kept its weight

    def test_upgrade_is_a_no_op_once_done(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _v2_store(db_path)
        upgrade_store(db_path)
        assert upgrade_store(db_path) is False

    def test_current_store_is_untouched(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert upgrade_store(db_path) is False
        assert load_embedding_matrix(db_path)[0] == ["uid000001", "uid000002"]


class TestStoreFileMode:
    def test_a_built_store_is_owner_only(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

    def test_a_world_readable_store_is_tightened_on_next_use(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        db_path.chmod(0o644)
        upgrade_store(db_path)
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
