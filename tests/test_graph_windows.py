from pathlib import Path

import pytest

from roam_semantic_search import graph_windows
from roam_semantic_search.graph_windows import WindowState, open_graph_names, window_state_for

_ONE_WINDOW: str = (
    "{:roam/windows [{:window/bounds {:x 135, :y 30, :width 1158, :height 1158},"
    ' :window/url "https://roamresearch.com/?server-port=3333#/app/hippo"}],'
    " :roam/local-api-enabled? true}"
)

_TWO_WINDOWS: str = (
    "{:roam/windows [{:window/bounds {:x 0, :y 0, :width 100, :height 100},"
    ' :window/url "https://roamresearch.com/?server-port=3333#/app/hippo"}'
    " {:window/bounds {:x 1, :y 1, :width 100, :height 100},"
    ' :window/url "https://roamresearch.com/?server-port=3333#/app/SCFH/page/abc123"}],'
    " :roam/local-api-enabled? true}"
)


@pytest.fixture
def state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path: Path = tmp_path / "user-config.edn"
    monkeypatch.setattr(graph_windows, "WINDOW_STATE_PATH", path)
    return path


class TestOpenGraphNames:
    def test_reads_the_single_open_window(self, state_file: Path) -> None:
        state_file.write_text(_ONE_WINDOW, encoding="utf-8")
        assert open_graph_names() == ("hippo",)

    def test_reads_every_window_in_file_order(self, state_file: Path) -> None:
        state_file.write_text(_TWO_WINDOWS, encoding="utf-8")
        assert open_graph_names() == ("hippo", "SCFH")

    def test_ignores_a_roam_url_outside_the_windows_vector(self, state_file: Path) -> None:
        state_file.write_text(
            _ONE_WINDOW.rstrip("}") + ' :roam/last-visited "https://roamresearch.com/#/app/Apple"}',
            encoding="utf-8",
        )
        assert open_graph_names() == ("hippo",)

    def test_percent_decodes_a_graph_name(self, state_file: Path) -> None:
        state_file.write_text(_ONE_WINDOW.replace("#/app/hippo", "#/app/my%20graph"), encoding="utf-8")
        assert open_graph_names() == ("my graph",)

    def test_no_open_window_is_an_empty_tuple(self, state_file: Path) -> None:
        state_file.write_text("{:roam/windows [], :roam/local-api-enabled? true}", encoding="utf-8")
        assert open_graph_names() == ()

    def test_absent_file_is_unknown(self, state_file: Path) -> None:
        assert not state_file.exists()
        assert open_graph_names() is None

    def test_file_without_the_windows_key_is_unknown(self, state_file: Path) -> None:
        state_file.write_text("{:roam/local-api-enabled? true}", encoding="utf-8")
        assert open_graph_names() is None


class TestWindowStateFor:
    def test_open_window_reports_open(self, state_file: Path) -> None:
        state_file.write_text(_ONE_WINDOW, encoding="utf-8")
        assert window_state_for("hippo") is WindowState.OPEN

    def test_graph_name_matches_case_insensitively(self, state_file: Path) -> None:
        state_file.write_text(_TWO_WINDOWS, encoding="utf-8")
        assert window_state_for("scfh") is WindowState.OPEN

    def test_absent_graph_reports_closed(self, state_file: Path) -> None:
        state_file.write_text(_ONE_WINDOW, encoding="utf-8")
        assert window_state_for("SCFH") is WindowState.CLOSED

    def test_unreadable_state_reports_unknown(self, state_file: Path) -> None:
        assert window_state_for("SCFH") is WindowState.UNKNOWN

    def test_closed_is_distinct_from_unknown(self, state_file: Path) -> None:
        state_file.write_text("{:roam/windows [], :roam/local-api-enabled? true}", encoding="utf-8")
        assert window_state_for("SCFH") is WindowState.CLOSED
