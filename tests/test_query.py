"""Tests for roam_semantic_search.query."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from roam_semantic_search import query
from roam_semantic_search.normalize import IndexRecord, PdfLocator, RecordKind
from roam_semantic_search.pdf_text import Media
from roam_semantic_search.query import SearchHit, collapse_key, rrf_fused, search_store
from roam_semantic_search.store import StoreMeta, write_store


class TestRrfFused:
    def test_item_in_both_rankings_beats_single_ranking_leader(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a", "b"], ["b", "c"]])
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_rank_order_within_one_ranking(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a", "b", "c"]])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_scores_are_reciprocal_rank_sums(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a"], ["a"]], rrf_k=60)
        assert scores["a"] == 2.0 / 61.0

    def test_empty_rankings_fuse_to_nothing(self) -> None:
        assert rrf_fused([]) == {}
        assert rrf_fused([[], []]) == {}


class TestCollapseKey:
    def test_page_and_block_records_collapse_by_themselves(self) -> None:
        assert collapse_key("uid000001") == "uid000001"

    def test_image_chunks_collapse_to_their_image(self) -> None:
        assert collapse_key("uid000001/img/abcd1234/1.2") == "uid000001/img/abcd1234"

    def test_pdf_chunks_collapse_to_their_pdf(self) -> None:
        assert collapse_key("uid000001/pdf/abcd1234/3.2") == "uid000001/pdf/abcd1234"
        assert collapse_key("uid000001/pdf/abcd1234/1.1") == collapse_key("uid000001/pdf/abcd1234/9.4")
        assert collapse_key("uid000001/pdf/ffff0000/1.1") != collapse_key("uid000001/pdf/abcd1234/1.1")


def _record(uid: str, text: str, pdf: PdfLocator | None = None) -> IndexRecord:
    return IndexRecord(
        uid=uid,
        page_title="Garden",
        breadcrumb="Garden",
        text=text,
        concepts=(),
        tags=(),
        descendant_text="",
        embed_input=text,
        content_hash="0" * 64,
        edited_at=1000,
        is_page=False,
        pdf=pdf,
    )


class TestSearchStoreCollapse:
    def test_one_hit_per_pdf_at_its_best_chunk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def embed_query(query_text: str, model: str, base_url: str) -> Sequence[float]:
            return [1.0, 0.0, 0.0]

        monkeypatch.setattr(query, "embed_query", embed_query)
        records: Final[list[IndexRecord]] = [
            _record("uid000001", "notes on the shed warranty"),
            _record("uid000002", "warranty page one", PdfLocator(key="abcd1234", page=1, chunk=1, name="shed.pdf")),
            _record("uid000002", "warranty page two", PdfLocator(key="abcd1234", page=2, chunk=1, name="shed.pdf")),
            _record("uid000002", "warranty page three", PdfLocator(key="abcd1234", page=3, chunk=1, name="shed.pdf")),
        ]
        vectors = np.asarray([[0.5, 0.5, 0.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-11-13T00:00:00+00:00", record_count=4
        )
        db_path: Final[Path] = tmp_path / "test.db"
        write_store(db_path, records, vectors, meta)

        hits: Final[list[SearchHit]] = search_store(db_path, "warranty", k=10)
        pdf_hits: Final[list[SearchHit]] = [hit for hit in hits if hit.kind is RecordKind.PDF]
        assert len(hits) == 2
        assert len(pdf_hits) == 1
        assert (pdf_hits[0].uid, pdf_hits[0].pdf_page, pdf_hits[0].pdf_name) == ("uid000002", 2, "shed.pdf")
        assert (pdf_hits[0].attachment_name, pdf_hits[0].attachment_page) == ("shed.pdf", 2)
        assert [hit.kind for hit in hits if hit.uid == "uid000001"] == [RecordKind.BLOCK]


class TestImageHits:
    def test_an_image_hit_names_its_attachment_but_carries_no_pdf_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def embed_query(query_text: str, model: str, base_url: str) -> Sequence[float]:
            return [1.0, 0.0, 0.0]

        monkeypatch.setattr(query, "embed_query", embed_query)
        locator: Final[PdfLocator] = PdfLocator(key="abcd1234", page=1, chunk=1, name="layout.svg", media=Media.SVG)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-11-14T00:00:00+00:00", record_count=1
        )
        db_path: Final[Path] = tmp_path / "test.db"
        write_store(
            db_path,
            [_record("uid000003", "raised bed layout", locator)],
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            meta,
        )
        hit: Final[SearchHit] = search_store(db_path, "layout", k=5)[0]
        assert (hit.kind, hit.uid, hit.attachment_name) == (RecordKind.IMAGE, "uid000003", "layout.svg")
        assert (hit.attachment_page, hit.pdf_name, hit.pdf_page) == (None, None, None)
