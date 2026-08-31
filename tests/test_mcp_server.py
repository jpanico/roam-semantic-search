"""Tests for the MCP server's staleness judgment."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

from roam_semantic_search import mcp_server
from roam_semantic_search.graph_registry import RegisteredGraph
from roam_semantic_search.graph_windows import WindowState
from roam_semantic_search.mcp_server import AutoRefresh, RefreshOutcome, _require_open_window, index_age_seconds
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


class TestRequireOpenWindow:
    """_require_open_window: the environment's open-window requirement, read truthily."""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " on "])
    def test_truthy_spellings_require_an_open_window(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """The documented truthy spellings all turn the requirement on."""
        monkeypatch.setenv("ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW", raw)
        assert _require_open_window() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", ""])
    def test_other_values_leave_it_off(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anything else leaves auto-refresh ungated, as it was before the flag existed."""
        monkeypatch.setenv("ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW", raw)
        assert _require_open_window() is False

    def test_unset_leaves_it_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset variable preserves the behaviour that predates the flag."""
        monkeypatch.delenv("ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW", raising=False)
        assert _require_open_window() is False


def stub_auto_refresh(monkeypatch: pytest.MonkeyPatch, state: WindowState, required: bool) -> list[str]:
    """Stand a stale store in front of a recording refresh; report the graphs it reached.

    Args:
        monkeypatch: The patcher to install the stubs through.
        state: The window state the graph should appear to be in.
        required: Whether the environment demands an open window.

    Returns:
        The list a stubbed refresh appends each reached graph name to.
    """
    reached: Final[list[str]] = []

    def read_meta(db_path: Path) -> StoreMeta:
        return meta_with("2020-01-01T00:00:00+00:00", None)

    def resolve_graph(selector: str) -> RegisteredGraph:
        return RegisteredGraph(name=selector)

    def window_state_for(graph_name: str) -> WindowState:
        return state

    def api_endpoint_for(graph: RegisteredGraph) -> str:
        return graph.name

    def refresh_store(db_path: Path, endpoint: str, **kwargs: object) -> None:
        reached.append(endpoint)

    monkeypatch.setenv("ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW", "1" if required else "0")
    monkeypatch.setattr(mcp_server, "read_meta", read_meta)
    monkeypatch.setattr(mcp_server, "resolve_graph", resolve_graph)
    monkeypatch.setattr(mcp_server, "window_state_for", window_state_for)
    monkeypatch.setattr(mcp_server, "api_endpoint_for", api_endpoint_for)
    monkeypatch.setattr(mcp_server, "refresh_store", refresh_store)
    return reached


class TestEnsuredFreshWindowGate:
    """_ensured_fresh: whether a stale index is refreshed when its graph has no open window."""

    def test_closed_window_skips_the_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: an automatic refresh never provokes an unlock prompt."""
        reached: Final[list[str]] = stub_auto_refresh(monkeypatch, WindowState.CLOSED, required=True)
        outcome: Final[AutoRefresh] = mcp_server._ensured_fresh("SCFH", Path("ignored.db"))
        assert outcome.outcome is RefreshOutcome.WINDOW_CLOSED
        assert reached == []

    def test_unknown_window_state_also_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a confirmed-open window satisfies the requirement."""
        reached: Final[list[str]] = stub_auto_refresh(monkeypatch, WindowState.UNKNOWN, required=True)
        outcome: Final[AutoRefresh] = mcp_server._ensured_fresh("SCFH", Path("ignored.db"))
        assert outcome.outcome is RefreshOutcome.WINDOW_CLOSED
        assert reached == []

    def test_open_window_refreshes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An open graph is refreshed exactly as before."""
        reached: Final[list[str]] = stub_auto_refresh(monkeypatch, WindowState.OPEN, required=True)
        outcome: Final[AutoRefresh] = mcp_server._ensured_fresh("hippo", Path("ignored.db"))
        assert outcome.outcome is RefreshOutcome.REFRESHED
        assert reached == ["hippo"]

    def test_requirement_off_refreshes_a_closed_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the requirement the window state is not consulted at all."""
        reached: Final[list[str]] = stub_auto_refresh(monkeypatch, WindowState.CLOSED, required=False)
        outcome: Final[AutoRefresh] = mcp_server._ensured_fresh("SCFH", Path("ignored.db"))
        assert outcome.outcome is RefreshOutcome.REFRESHED
        assert reached == ["SCFH"]
