"""Tests for roam_semantic_search.refresh."""

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from guffin.roam.local_api import ApiEndpoint

from roam_semantic_search.normalize import IndexRecord
from roam_semantic_search.refresh import RefreshPlan, refresh_plan, refresh_store
from roam_semantic_search.store import StoreMeta, write_store


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
        assert plan.deleted_uids == ()
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
        assert plan.deleted_uids == ("uid000009",)

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
