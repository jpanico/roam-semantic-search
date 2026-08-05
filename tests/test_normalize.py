"""Tests for roam_semantic_search.normalize."""

from typing import Final

from roam_semantic_search.normalize import IndexRecord, _page_ref_names, _plain_text, normalized_records


class TestPlainText:
    def test_page_refs_unwrap_nested(self) -> None:
        assert _plain_text("see [[AI as a [[Ghost]] figure]] now", {}) == "see AI as a Ghost figure now"

    def test_hashtag_forms_unwrap(self) -> None:
        assert _plain_text("#tag and #[[long tag]]", {}) == "tag and long tag"

    def test_color_tag_drops(self) -> None:
        assert _plain_text("#c:red important", {}) == "important"

    def test_todo_marker_drops(self) -> None:
        assert _plain_text("{{[[TODO]]}} buy milk", {}) == "buy milk"

    def test_widget_drops(self) -> None:
        assert _plain_text("{{[[table]]}} after", {}) == "after"

    def test_image_keeps_alt(self) -> None:
        assert _plain_text("![a hippo](https://example.com/x.png)", {}) == "a hippo"

    def test_link_keeps_display(self) -> None:
        assert _plain_text("[the docs](https://example.com)", {}) == "the docs"

    def test_block_ref_resolves_through_map(self) -> None:
        assert _plain_text("as ((abcdefghi)) says", {"abcdefghi": "the target"}) == "as the target says"

    def test_block_ref_without_target_drops(self) -> None:
        assert _plain_text("as ((abcdefghi)) says", {}) == "as says"

    def test_embed_resolves_as_reference(self) -> None:
        raw: Final[str] = "{{[[embed]]: ((abcdefghi))}}"
        assert _plain_text(raw, {"abcdefghi": "embedded text"}) == "embedded text"

    def test_styling_delimiters_drop(self) -> None:
        assert _plain_text("**bold** __italic__ ^^mark^^ ~~gone~~", {}) == "bold italic mark gone"

    def test_attribute_head_becomes_colon(self) -> None:
        assert _plain_text("status:: draft", {}) == "status: draft"

    def test_code_fence_drops_keeping_code(self) -> None:
        assert _plain_text("```python\nx = 1\n```", {}) == "x = 1"

    def test_whitespace_collapses(self) -> None:
        assert _plain_text("a\n\n  b\tc", {}) == "a b c"


def _page_row(dbid: int, uid: str, title: str, child_ids: list[int] | None = None) -> dict[str, object]:
    row: Final[dict[str, object]] = {"id": dbid, "uid": uid, "title": title, "time": 1000}
    if child_ids is not None:
        row["children"] = [{"id": cid} for cid in child_ids]
    return row


def _block_row(
    dbid: int,
    uid: str,
    string: str,
    page_id: int,
    parent_ids: list[int],
    child_ids: list[int] | None = None,
    order: int = 0,
) -> dict[str, object]:
    row: Final[dict[str, object]] = {
        "id": dbid,
        "uid": uid,
        "string": string,
        "page": {"id": page_id},
        "parents": [{"id": pid} for pid in parent_ids],
        "order": order,
        "time": 2000,
        "edit-time": 3000,
    }
    if child_ids is not None:
        row["children"] = [{"id": cid} for cid in child_ids]
    return row


class TestNormalizedRecords:
    def test_page_and_block_records(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "My Page"),
            _block_row(2, "block0001", "top level", 1, [1]),
            _block_row(3, "block0002", "a child", 1, [1, 2]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        by_uid: Final[dict[str, IndexRecord]] = {record.uid: record for record in records}
        assert by_uid["page00001"].is_page
        assert by_uid["block0001"].breadcrumb == "My Page"
        assert by_uid["block0001"].embed_input == "My Page › top level"
        assert by_uid["block0002"].breadcrumb == "My Page › top level"
        assert by_uid["block0002"].edited_at == 3000

    def test_page_title_markup_cleaned_for_embedding_kept_for_display(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "The new [[Programmer]] and [[AI]]"),
            _block_row(2, "block0001", "a claim", 1, [1]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        by_uid: Final[dict[str, IndexRecord]] = {record.uid: record for record in records}
        assert by_uid["page00001"].page_title == "The new [[Programmer]] and [[AI]]"
        assert by_uid["page00001"].embed_input == "The new Programmer and AI | concepts: Programmer · AI"
        assert by_uid["block0001"].embed_input == "The new Programmer and AI › a claim"

    def test_breadcrumb_orders_ancestors_root_first(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P"),
            _block_row(2, "block0001", "grandparent", 1, [1]),
            _block_row(3, "block0002", "parent", 1, [1, 2]),
            # Stub order deliberately scrambled: depth ordering must not rely on wire order.
            _block_row(4, "block0003", "leaf", 1, [3, 1, 2]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        leaf: Final[IndexRecord] = next(record for record in records if record.uid == "block0003")
        assert leaf.breadcrumb == "P › grandparent · parent"

    def test_roam_system_pages_skipped_with_blocks(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "roam/css"),
            _block_row(2, "block0001", "some css", 1, [1]),
        ]
        assert normalized_records(rows) == []

    def test_daily_notes_included_by_default_skipped_on_request(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "08-04-2026", "August 4th, 2026"),
            _block_row(2, "block0001", "journal entry", 1, [1]),
        ]
        included: Final[list[IndexRecord]] = normalized_records(rows)
        assert {record.uid for record in included} == {"08-04-2026", "block0001"}
        assert normalized_records(rows, include_daily_notes=False) == []

    def test_empty_after_cleanup_skipped(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P"),
            _block_row(2, "block0001", "{{[[table]]}}", 1, [1]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        assert [record.uid for record in records] == ["page00001"]

    def test_block_ref_resolves_one_level(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P"),
            _block_row(2, "block0001", "the source claim", 1, [1]),
            _block_row(3, "block0002", "see ((block0001))", 1, [1]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        referring: Final[IndexRecord] = next(record for record in records if record.uid == "block0002")
        assert referring.text == "see the source claim"


class TestPageRefNames:
    def test_nested_reference_yields_every_level_innermost_first(self) -> None:
        assert _page_ref_names("see [[[[Illustration]] Brief]]", {}) == ("Illustration", "Illustration Brief")

    def test_hashtag_spellings_count_as_references(self) -> None:
        assert _page_ref_names("#Emi and #[[Hedge Maze]]", {}) == ("Hedge Maze", "Emi")

    def test_attribute_head_and_color_tags_are_not_references(self) -> None:
        assert _page_ref_names("tags:: #c:red #Illustration", {}) == ("Illustration",)

    def test_names_deduplicate_preserving_order(self) -> None:
        assert _page_ref_names("[[A]] then [[B]] then [[A]] then #A", {}) == ("A", "B")

    def test_plain_text_yields_no_names(self) -> None:
        assert _page_ref_names("no references here", {}) == ()


class TestRetrievalEmphasis:
    def test_tags_child_folds_into_parent(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P", child_ids=[2]),
            _block_row(2, "block0001", "sketches from [[Emi]]", 1, [1], child_ids=[3, 4]),
            _block_row(3, "block0002", "a regular child", 1, [1, 2], order=0),
            _block_row(4, "block0003", "tags:: #Illustration", 1, [1, 2], order=1),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        parent: Final[IndexRecord] = next(record for record in records if record.uid == "block0001")
        assert parent.concepts == ("Emi",)
        assert parent.tags == ("Illustration",)
        assert parent.descendant_text == "a regular child · tags: Illustration"
        assert parent.embed_input == (
            "P › sketches from Emi | concepts: Emi | tags: Illustration | a regular child · tags: Illustration"
        )

    def test_tags_plain_values_fall_back_to_comma_split(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P", child_ids=[2]),
            _block_row(2, "block0001", "a parent", 1, [1], child_ids=[3]),
            _block_row(3, "block0002", "tags:: alpha, beta", 1, [1, 2]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        parent: Final[IndexRecord] = next(record for record in records if record.uid == "block0001")
        assert parent.tags == ("alpha", "beta")

    def test_descendant_text_is_depth_first_in_document_order(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P", child_ids=[2]),
            # Children stub order deliberately scrambled: document order comes from each child's order field.
            _block_row(2, "block0001", "root", 1, [1], child_ids=[4, 3], order=0),
            _block_row(3, "block0002", "second child", 1, [1, 2], child_ids=[5], order=1),
            _block_row(4, "block0003", "first child", 1, [1, 2], order=0),
            _block_row(5, "block0004", "grandchild", 1, [1, 2, 3], order=0),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        root: Final[IndexRecord] = next(record for record in records if record.uid == "block0001")
        assert root.descendant_text == "first child · second child · grandchild"

    def test_page_record_carries_content_and_classification(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "[[Illustration]] thumbnails", child_ids=[2, 3]),
            _block_row(2, "block0001", "tags:: #Illustration", 1, [1], order=0),
            _block_row(3, "block0002", "sketch 1", 1, [1], order=1),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        page: Final[IndexRecord] = next(record for record in records if record.uid == "page00001")
        assert page.concepts == ("Illustration",)
        assert page.tags == ("Illustration",)
        assert page.descendant_text == "tags: Illustration · sketch 1"

    def test_embed_cap_cuts_descendants_before_emphasis(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            _page_row(1, "page00001", "P", child_ids=[2]),
            _block_row(2, "block0001", "about [[Emi]]", 1, [1], child_ids=[3]),
            _block_row(3, "block0002", "x" * 9000, 1, [1, 2]),
        ]
        records: Final[list[IndexRecord]] = normalized_records(rows)
        parent: Final[IndexRecord] = next(record for record in records if record.uid == "block0001")
        assert len(parent.embed_input) == 8000
        assert "concepts: Emi" in parent.embed_input
