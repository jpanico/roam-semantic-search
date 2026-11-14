"""Tests for the CLI's open-window gate (what a scheduled run does when the graph is closed) and extract-pdfs."""

from pathlib import Path
from typing import Final

import pytest
from guffin.roam.local_api import ApiEndpoint
from typer.testing import CliRunner

from roam_semantic_search import cli
from roam_semantic_search.graph_windows import WindowState
from roam_semantic_search.pdf_text import ExtractionSummary, PdfStatus
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


@pytest.fixture
def pdf_run(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub extract-pdfs' Local API reach; records each graph whose endpoint was built."""
    endpoints: Final[list[str]] = []

    def api_endpoint(selector: str, port: int | None, token: str | None) -> ApiEndpoint:
        endpoints.append(selector)
        return ApiEndpoint.from_parts(local_api_port=1, graph_name=selector, bearer_token="unused")

    def fetch_graph(endpoint: ApiEndpoint) -> list[dict[str, object]]:
        return []

    monkeypatch.setattr(cli, "_api_endpoint", api_endpoint)
    monkeypatch.setattr(cli, "fetch_graph", fetch_graph)
    return endpoints


class TestExtractPdfs:
    """extract-pdfs: the open-window gate, the summary line, and the abort exit."""

    def test_closed_window_skips_without_touching_the_api(
        self, tmp_path: Path, pdf_run: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_window_state(monkeypatch, WindowState.CLOSED)
        result = runner.invoke(
            cli.app,
            ["extract-pdfs", "--graph", "SCFH", "--pdf-cache", str(tmp_path / "c.db"), "--require-open-window"],
        )
        assert result.exit_code == 0
        assert "skipped" in result.output
        assert pdf_run == []

    def test_reports_the_run(self, tmp_path: Path, pdf_run: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
        def extract_pdf_texts(*args: object, **kwargs: object) -> ExtractionSummary:
            return ExtractionSummary(
                distinct_count=3, current_count=1, outcomes={PdfStatus.OK: 1, PdfStatus.NO_TEXT: 1}, pending_count=0
            )

        monkeypatch.setattr(cli, "extract_pdf_texts", extract_pdf_texts)
        result = runner.invoke(cli.app, ["extract-pdfs", "--graph", "SCFH", "--pdf-cache", str(tmp_path / "c.db")])
        assert result.exit_code == 0
        assert "extracted 2" in result.output
        assert "1 no-text, 1 ok" in result.output

    def test_a_failing_graph_aborts_with_exit_1(
        self, tmp_path: Path, pdf_run: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def extract_pdf_texts(*args: object, **kwargs: object) -> ExtractionSummary:
            raise RuntimeError("5 PDF fetches failed in a row")

        monkeypatch.setattr(cli, "extract_pdf_texts", extract_pdf_texts)
        result = runner.invoke(cli.app, ["extract-pdfs", "--graph", "SCFH", "--pdf-cache", str(tmp_path / "c.db")])
        assert result.exit_code == 1
        assert "aborted" in result.output
