"""Tests for roam_semantic_search.store."""

from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from roam_semantic_search.normalize import IndexRecord
from roam_semantic_search.store import (
    SCHEMA_VERSION,
    StoreMeta,
    delete_records,
    keyword_ranked_uids,
    load_embedding_matrix,
    read_meta,
    records_by_uid,
    stamp_refresh,
    stored_hashes,
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
        assert keyword_ranked_uids(db_path, "hippo", 10) == ["uid000001"]

    def test_keyword_search_survives_fts_syntax(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert keyword_ranked_uids(db_path, 'hippo AND "NEAR(', 10) == ["uid000001"]

    def test_schema_version_stamped(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        assert read_meta(db_path).schema_version == SCHEMA_VERSION

    def test_records_read_back(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        records = records_by_uid(db_path, ["uid000002", "missing00"])
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
        assert keyword_ranked_uids(db_path, "hippo", 10) == ["uid0conc1", "uid00tag1", "uid0text1"]

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
        assert keyword_ranked_uids(db_path, "hippo", 10) == ["uid0desc1"]


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
        assert keyword_ranked_uids(db_path, "whale", 10) == ["uid000003"]
        assert keyword_ranked_uids(db_path, "swims", 10) == []  # replaced text left the FTS mirror

    def test_delete_removes_record_and_fts_row(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        _built_store(db_path)
        delete_records(db_path, ["uid000001", "missing00"])
        uids, _ = load_embedding_matrix(db_path)
        assert uids == ["uid000002"]
        assert keyword_ranked_uids(db_path, "hippo", 10) == []

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
