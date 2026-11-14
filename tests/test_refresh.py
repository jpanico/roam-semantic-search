"""Tests for roam_semantic_search.refresh."""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from guffin.roam.local_api import ApiEndpoint

from roam_semantic_search import refresh
from roam_semantic_search.normalize import IndexRecord, PdfLocator, normalized_records
from roam_semantic_search.pdf_text import ExtractionSummary, PdfEmbed, PdfFetchBudget, PdfStatus, PdfText, url_key
from roam_semantic_search.refresh import RefreshPlan, RefreshSummary, budgeted_embeds, refresh_plan, refresh_store
from roam_semantic_search.store import StoreMeta, stored_record_count, write_store


def _record(uid: str, content_hash: str) -> IndexRecord:
    return IndexRecord(
        uid=uid,
        page_title="P",
        breadcrumb="P",
        text=f"text of {uid}",
        concepts=(),
        tags=(),
        descendant_text="",
        embed_input=f"P › text of {uid}",
        content_hash=content_hash,
        edited_at=1000,
        is_page=False,
    )


class TestRefreshPlan:
    def test_unchanged_records_selected_by_matching_hash(self) -> None:
        records: Final[list[IndexRecord]] = [_record("uid000001", "aaa"), _record("uid000002", "bbb")]
        plan: Final[RefreshPlan] = refresh_plan(records, {"uid000001": "aaa", "uid000002": "bbb"})
        assert plan.to_embed == ()
        assert plan.deleted_ids == ()
        assert plan.unchanged_count == 2

    def test_changed_hash_selects_reembedding(self) -> None:
        records: Final[list[IndexRecord]] = [_record("uid000001", "aaa"), _record("uid000002", "CHANGED")]
        plan: Final[RefreshPlan] = refresh_plan(records, {"uid000001": "aaa", "uid000002": "bbb"})
        assert [record.uid for record in plan.to_embed] == ["uid000002"]
        assert plan.unchanged_count == 1

    def test_new_record_selects_embedding(self) -> None:
        records: Final[list[IndexRecord]] = [_record("uid000001", "aaa"), _record("uid000003", "ccc")]
        plan: Final[RefreshPlan] = refresh_plan(records, {"uid000001": "aaa"})
        assert [record.uid for record in plan.to_embed] == ["uid000003"]

    def test_vanished_uid_selects_deletion(self) -> None:
        records: Final[list[IndexRecord]] = [_record("uid000001", "aaa")]
        plan: Final[RefreshPlan] = refresh_plan(records, {"uid000001": "aaa", "uid000009": "zzz"})
        assert plan.deleted_ids == ("uid000009",)

    def test_empty_store_embeds_everything(self) -> None:
        records: Final[list[IndexRecord]] = [_record("uid000001", "aaa"), _record("uid000002", "bbb")]
        plan: Final[RefreshPlan] = refresh_plan(records, {})
        assert len(plan.to_embed) == 2
        assert plan.unchanged_count == 0


class TestSchemaVersionGuard:
    def test_outdated_store_refuses_refresh(self, tmp_path: Path) -> None:
        db_path: Final[Path] = tmp_path / "test.db"
        outdated_meta: Final[StoreMeta] = StoreMeta(
            graph_name="G",
            embed_model="m",
            dimension=3,
            built_at="2026-08-05T00:00:00+00:00",
            record_count=1,
            schema_version=1,
        )
        write_store(db_path, [_record("uid000001", "aaa")], np.zeros((1, 3), dtype=np.float32), outdated_meta)
        endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(local_api_port=1, graph_name="G", bearer_token="unused")
        with pytest.raises(ValueError, match="schema"):
            refresh_store(db_path, endpoint)


def _chunk(uid: str, page: int, content_hash: str) -> IndexRecord:
    return _record(uid, content_hash).model_copy(update={"pdf": PdfLocator(key="abcd1234", page=page, chunk=1)})


class TestPdfChunkPlans:
    def test_chunks_of_one_block_diff_independently(self) -> None:
        host: Final[IndexRecord] = _record("uid000001", "hhh")
        one: Final[IndexRecord] = _chunk("uid000001", 1, "aaa")
        two: Final[IndexRecord] = _chunk("uid000001", 2, "bbb")
        plan: Final[RefreshPlan] = refresh_plan(
            [host, one, two], {"uid000001": "hhh", one.record_id: "aaa", two.record_id: "old"}
        )
        assert [record.record_id for record in plan.to_embed] == ["uid000001/pdf/abcd1234/2.1"]
        assert plan.unchanged_count == 2

    def test_vanished_chunk_selects_deletion_without_touching_its_host(self) -> None:
        host: Final[IndexRecord] = _record("uid000001", "hhh")
        plan: Final[RefreshPlan] = refresh_plan([host], {"uid000001": "hhh", "uid000001/pdf/abcd1234/1.1": "aaa"})
        assert plan.deleted_ids == ("uid000001/pdf/abcd1234/1.1",)
        assert plan.to_embed == ()


class TestBudgetedEmbeds:
    def test_blocks_always_embed_and_pdf_chunks_are_capped_in_order(self) -> None:
        records: Final[list[IndexRecord]] = [
            _chunk("uid000001", 1, "a"),
            _record("uid000002", "b"),
            _chunk("uid000001", 2, "c"),
            _chunk("uid000001", 3, "d"),
            _record("uid000003", "e"),
        ]
        now, deferred = budgeted_embeds(records, 1)
        assert [record.record_id for record in now] == ["uid000001/pdf/abcd1234/1.1", "uid000002", "uid000003"]
        assert [record.record_id for record in deferred] == [
            "uid000001/pdf/abcd1234/2.1",
            "uid000001/pdf/abcd1234/3.1",
        ]

    def test_no_cap_embeds_everything(self) -> None:
        records: Final[list[IndexRecord]] = [_chunk("uid000001", 1, "a"), _record("uid000002", "b")]
        assert budgeted_embeds(records, None) == (records, [])


_PDF_URL: Final[str] = (
    "https://firebasestorage.googleapis.com/v0/b/test.appspot.com/o/imgs%2Fapp%2Ftest%2FAb3dE.pdf?alt=media&token=t"
)

_ROWS: Final[list[dict[str, object]]] = [
    {"id": 1, "uid": "page00001", "title": "Garden", "time": 1000},
    {
        "id": 2,
        "uid": "blk000001",
        "string": f"manual: {{{{pdf: {_PDF_URL}}}}}",
        "page": {"id": 1},
        "parents": [{"id": 1}],
        "order": 0,
        "time": 2000,
    },
    {
        "id": 3,
        "uid": "blk000002",
        "string": "plant the tomatoes",
        "page": {"id": 1},
        "parents": [{"id": 1}],
        "order": 1,
        "time": 2000,
    },
]

_PDF_TEXTS: Final[dict[str, PdfText]] = {
    url_key(_PDF_URL): PdfText(
        status=PdfStatus.OK,
        extractor_version=1,
        upload_name="shed.pdf",
        page_texts=("Invoice for the garden shed", "Warranty covers the roof panels", "Assembly takes two hours"),
        extracted_at="2026-11-14T00:00:00+00:00",
    )
}


class TestRefreshStorePdfBudget:
    @pytest.fixture
    def store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A v3 store holding the graph's page and block records, with fetching and embedding stubbed."""
        db_path: Final[Path] = tmp_path / "G.db"
        records: Final[list[IndexRecord]] = normalized_records(_ROWS)
        meta: Final[StoreMeta] = StoreMeta(
            graph_name="G", embed_model="m", dimension=3, built_at="2026-11-14T00:00:00+00:00", record_count=2
        )
        write_store(db_path, records, np.ones((len(records), 3), dtype=np.float32), meta)

        def fetch_graph(endpoint: ApiEndpoint) -> list[dict[str, object]]:
            return _ROWS

        def embed_texts(texts: Sequence[str], model: str, base_url: str) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

        def cached_pdf_texts(cache_path: Path) -> dict[str, PdfText]:
            return _PDF_TEXTS

        monkeypatch.setattr(refresh, "fetch_graph", fetch_graph)
        monkeypatch.setattr(refresh, "embed_texts", embed_texts)
        monkeypatch.setattr(refresh, "cached_pdf_texts", cached_pdf_texts)
        return db_path

    def _refresh(self, store: Path, **kwargs: object) -> RefreshSummary:
        endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(local_api_port=1, graph_name="G", bearer_token="unused")
        return refresh_store(store, endpoint, pdf_cache_path=store.with_suffix(".pdf-text.db"), **kwargs)  # type: ignore[arg-type]

    def test_deferred_chunks_land_on_the_next_refresh(self, store: Path) -> None:
        first: Final[RefreshSummary] = self._refresh(store, max_pdf_embeds=2)
        assert (first.embedded_count, first.embed_deferred_count) == (2, 1)
        assert first.record_count == stored_record_count(store) == len(normalized_records(_ROWS)) + 2
        second: Final[RefreshSummary] = self._refresh(store, max_pdf_embeds=2)
        assert (second.embedded_count, second.embed_deferred_count, second.deleted_count) == (1, 0, 0)
        assert second.record_count == len(normalized_records(_ROWS)) + 3

    def test_without_a_fetch_budget_nothing_is_fetched(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: Final[list[object]] = []

        def extract_pdf_texts(*args: object, **kwargs: object) -> None:
            calls.append(args)

        monkeypatch.setattr(refresh, "extract_pdf_texts", extract_pdf_texts)
        summary: Final[RefreshSummary] = self._refresh(store)
        assert calls == []
        assert summary.pdf_fetched_count == 0

    def test_a_failed_pdf_fetch_does_not_fail_the_refresh(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def extract_pdf_texts(*args: object, **kwargs: object) -> None:
            raise RuntimeError("5 PDF fetches failed in a row")

        monkeypatch.setattr(refresh, "extract_pdf_texts", extract_pdf_texts)
        summary: Final[RefreshSummary] = self._refresh(store, pdf_fetch_budget=PdfFetchBudget(max_pdfs=10))
        assert summary.pdf_error is not None and "in a row" in summary.pdf_error
        assert summary.embedded_count == 3  # the cached PDF text was still indexed

    def test_the_fetch_covers_only_indexable_pages(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: Final[list[list[str]]] = []

        def extract_pdf_texts(
            cache: Path, embeds: Sequence[PdfEmbed], *args: object, **kwargs: object
        ) -> ExtractionSummary:
            seen.append([embed.host_uid for embed in embeds])
            return ExtractionSummary(distinct_count=1, current_count=0, outcomes={PdfStatus.OK: 1}, pending_count=4)

        monkeypatch.setattr(refresh, "extract_pdf_texts", extract_pdf_texts)
        summary: Final[RefreshSummary] = self._refresh(store, pdf_fetch_budget=PdfFetchBudget(max_pdfs=1))
        assert seen == [["blk000001"]]
        assert (summary.pdf_fetched_count, summary.pdf_pending_count) == (1, 4)
