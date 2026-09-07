"""Tests for the CLI's open-window gate: what a scheduled refresh does when the graph is closed."""

from pathlib import Path
from typing import Final

import pytest
from typer.testing import CliRunner

from roam_semantic_search import cli
from roam_semantic_search.graph_windows import WindowState
from roam_semantic_search.refresh import RefreshSummary

runner: Final[CliRunner] = CliRunner()

SUMMARY: Final[RefreshSummary] = RefreshSummary(embedded_count=0, deleted_count=0, unchanged_count=1, record_count=1)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An existing index file, so the refresh command gets past its store check."""
    path: Path = tmp_path / "SCFH.db"
    path.touch()
    return path


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Records every store a refresh actually took to the Local API."""
    stores: Final[list[Path]] = []

    def refresh_store(db_path: Path, endpoint: str, **kwargs: object) -> RefreshSummary:
        stores.append(db_path)
        return SUMMARY

    def api_endpoint(selector: str, port: int | None, token: str | None) -> str:
        return "endpoint"

    monkeypatch.setattr(cli, "refresh_store", refresh_store)
    monkeypatch.setattr(cli, "_api_endpoint", api_endpoint)
    return stores


def stub_window_state(monkeypatch: pytest.MonkeyPatch, state: WindowState) -> None:
    """Make every graph appear to be in *state*."""

    def window_state_for(graph_name: str) -> WindowState:
        return state

    monkeypatch.setattr(cli, "window_state_for", window_state_for)


class TestRefreshOpenWindowGate:
    """refresh --require-open-window: skip the graphs Roam certainly cannot serve."""

    def test_closed_window_skips_without_touching_the_api(
        self, store: Path, reached: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A closed graph is skipped, and skipping is a success — a scheduler sees no failure."""
        stub_window_state(monkeypatch, WindowState.CLOSED)
        result = runner.invoke(cli.app, ["refresh", "--graph", "SCFH", "--db", str(store), "--require-open-window"])
        assert result.exit_code == 0
        assert "skipped" in result.output
        assert reached == []

    def test_unknown_window_state_also_skips(
        self, store: Path, reached: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreadable list is not a listing, so it cannot satisfy the requirement."""
        stub_window_state(monkeypatch, WindowState.UNKNOWN)
        result = runner.invoke(cli.app, ["refresh", "--graph", "SCFH", "--db", str(store), "--require-open-window"])
        assert result.exit_code == 0
        assert reached == []

    def test_open_window_refreshes(self, store: Path, reached: list[Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """An open graph refreshes exactly as it did before the flag existed."""
        stub_window_state(monkeypatch, WindowState.OPEN)
        result = runner.invoke(cli.app, ["refresh", "--graph", "SCFH", "--db", str(store), "--require-open-window"])
        assert result.exit_code == 0
        assert reached == [store]

    def test_without_the_flag_the_window_state_is_not_consulted(
        self, store: Path, reached: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flag is opt-in: an unflagged run behaves as it always has."""
        stub_window_state(monkeypatch, WindowState.CLOSED)
        result = runner.invoke(cli.app, ["refresh", "--graph", "SCFH", "--db", str(store)])
        assert result.exit_code == 0
        assert reached == [store]
