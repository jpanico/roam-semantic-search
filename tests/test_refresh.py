"""Tests for roam_semantic_search.refresh."""

from typing import Final

from roam_semantic_search.normalize import IndexRecord
from roam_semantic_search.refresh import RefreshPlan, refresh_plan


def _record(uid: str, content_hash: str) -> IndexRecord:
    return IndexRecord(
        uid=uid,
        page_title="P",
        breadcrumb="P",
        text=f"text of {uid}",
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
