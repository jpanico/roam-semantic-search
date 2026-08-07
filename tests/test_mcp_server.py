"""Tests for the MCP server's staleness judgment."""

from datetime import UTC, datetime
from typing import Final

import pytest

from roam_semantic_search.mcp_server import index_age_seconds
from roam_semantic_search.store import StoreMeta

NOW: Final[datetime] = datetime(2026, 8, 7, 18, 0, 0, tzinfo=UTC)


def meta_with(built_at: str, refreshed_at: str | None) -> StoreMeta:
    """A StoreMeta whose non-timestamp fields are irrelevant boilerplate."""
    return StoreMeta(
        graph_name="G",
        embed_model="m",
        dimension=8,
        built_at=built_at,
        record_count=1,
        refreshed_at=refreshed_at,
    )


class TestIndexAgeSeconds:
    """index_age_seconds: the store's capture age, judged at a supplied moment."""

    def test_refresh_moment_wins_over_build_moment(self) -> None:
        """A refreshed store is as old as its refresh, not its build."""
        meta: Final[StoreMeta] = meta_with("2026-08-01T00:00:00+00:00", "2026-08-07T17:30:00+00:00")
        assert index_age_seconds(meta, NOW) == pytest.approx(1800.0)

    def test_never_refreshed_store_ages_from_its_build(self) -> None:
        """With no refresh recorded, the build moment is the capture moment."""
        meta: Final[StoreMeta] = meta_with("2026-08-07T16:00:00+00:00", None)
        assert index_age_seconds(meta, NOW) == pytest.approx(7200.0)

    def test_naive_timestamp_is_taken_as_utc(self) -> None:
        """A stored timestamp without an offset is interpreted as UTC, not local time."""
        meta: Final[StoreMeta] = meta_with("2026-08-07T17:00:00", None)
        assert index_age_seconds(meta, NOW) == pytest.approx(3600.0)

    def test_future_capture_is_negative_age(self) -> None:
        """A capture claiming to postdate now yields a negative age rather than an error."""
        meta: Final[StoreMeta] = meta_with("2026-08-07T19:00:00+00:00", None)
        assert index_age_seconds(meta, NOW) == pytest.approx(-3600.0)

    def test_unparseable_timestamp_is_none(self) -> None:
        """A garbage capture moment reports as unknown (None), which callers treat as stale."""
        meta: Final[StoreMeta] = meta_with("not-a-moment", None)
        assert index_age_seconds(meta, NOW) is None

    def test_unparseable_refresh_falls_back_to_nothing_not_build(self) -> None:
        """The refresh moment, once present, is authoritative — a broken one is not silently
        replaced by the older build moment, which would understate staleness."""
        meta: Final[StoreMeta] = meta_with("2026-08-07T16:00:00+00:00", "garbage")
        assert index_age_seconds(meta, NOW) is None
