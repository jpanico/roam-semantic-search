"""Which graphs Roam Desktop currently has a window open on.

Roam Desktop records its open windows in an EDN file of its own, so that a relaunch can
restore them:

.. code-block:: clojure

    {:roam/windows [{:window/bounds {:x 135, :y 30, :width 1158, :height 1158},
                     :window/url "https://roamresearch.com/?server-port=3333#/app/hippo"}],
     :roam/local-api-enabled? true}

Each window names its graph in the ``#/app/<graph>`` fragment of its URL.  Reading that
file is the only way to learn which graphs are open **without touching the graphs**: the
Local API is addressed per graph (``/api/<name>``), it offers no graph-agnostic status
route, and a request naming a closed graph is itself what makes Roam Desktop open a window
for it — which, for an encrypted graph, is an unlock prompt awaiting a human.  A caller
that must not provoke that prompt therefore has to ask the filesystem, not the API.

The state is a snapshot of a file Roam owns and writes on its own schedule, so a reading
is evidence rather than proof; :class:`WindowState` keeps :data:`WindowState.UNKNOWN`
distinct from :data:`WindowState.CLOSED` so a caller can tell absence of evidence from
evidence of absence.

Public symbols:

- :data:`WINDOW_STATE_PATH` — where Roam records its open windows on this platform.
- :data:`GRAPH_URL_PATTERN` / :data:`GRAPH_URL_RE` — the ``#/app/<graph>`` fragment.
- :data:`WINDOWS_VECTOR_PATTERN` / :data:`WINDOWS_VECTOR_RE` — the ``:roam/windows`` vector.
- :class:`WindowState` — whether a graph has a window open, has none, or cannot be told.
- :func:`open_graph_names` — the graphs with a window open, or ``None`` when unreadable.
- :func:`window_state_for` — one graph's window state.
"""

import enum
import logging
import sys
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import regex
from pydantic import validate_call

logger = logging.getLogger(__name__)

_WINDOW_STATE_FILE_NAME: Final[str] = "user-config.edn"
_APP_SUPPORT_DIR_NAME: Final[str] = "Roam Research"


def _window_state_path() -> Path:
    """The window-state file's location for the running platform.

    Roam Desktop is an Electron application, so the file sits in the per-user application
    data directory Electron assigns the app.  Only the macOS location is verified against a
    live install; the others follow Electron's documented convention.
    """
    home: Final[Path] = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / _APP_SUPPORT_DIR_NAME / _WINDOW_STATE_FILE_NAME
    if sys.platform == "win32":
        return home / "AppData" / "Roaming" / _APP_SUPPORT_DIR_NAME / _WINDOW_STATE_FILE_NAME
    return home / ".config" / _APP_SUPPORT_DIR_NAME / _WINDOW_STATE_FILE_NAME


WINDOW_STATE_PATH: Final[Path] = _window_state_path()
"""Where Roam Desktop records the windows it has open."""

WINDOWS_VECTOR_PATTERN: Final[str] = r":roam/windows\s*(?P<windows>\[(?:[^\[\]]++|(?P>windows))*+\])"
"""The ``:roam/windows`` vector, matched with balanced brackets so nesting cannot truncate it.

Narrowing to this vector before reading URLs keeps an unrelated Roam URL elsewhere in the
file — a key Roam may add at any time — from being mistaken for an open window.
"""

WINDOWS_VECTOR_RE: Final[regex.Pattern[str]] = regex.compile(WINDOWS_VECTOR_PATTERN)

GRAPH_URL_PATTERN: Final[str] = r"#/app/(?P<graph>[^\"/?#\s]+)"
"""A window URL's graph fragment.

The graph is the one segment following ``#/app/``; a window opened on a specific page
carries further segments (``#/app/<graph>/page/<uid>``), which are no part of the name.
"""

GRAPH_URL_RE: Final[regex.Pattern[str]] = regex.compile(GRAPH_URL_PATTERN)


class WindowState(enum.StrEnum):
    """Whether a graph has a Roam Desktop window open.

    ``UNKNOWN`` is deliberately distinct from ``CLOSED``: a missing or unparseable
    window-state file says nothing about the graph, and a caller gating a side effect on
    this answer will usually want to treat the two differently.
    """

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@validate_call
def open_graph_names() -> tuple[str, ...] | None:
    """The graphs Roam Desktop currently has a window open on.

    Returns:
        One entry per open window, in the order the file lists them, percent-decoded and
        with duplicates preserved (a graph may be open in more than one window); an empty
        tuple when Roam records no open window, and ``None`` when the state cannot be
        read at all — an absent, unreadable, or unrecognizable file.
    """
    if not WINDOW_STATE_PATH.is_file():
        logger.debug("no Roam window state at %s", WINDOW_STATE_PATH)
        return None
    try:
        text: Final[str] = WINDOW_STATE_PATH.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        logger.debug("Roam window state at %s is unreadable", WINDOW_STATE_PATH)
        return None
    vector: Final[regex.Match[str] | None] = WINDOWS_VECTOR_RE.search(text)
    if vector is None:
        logger.debug("Roam window state at %s records no :roam/windows vector", WINDOW_STATE_PATH)
        return None
    return tuple(unquote(match.group("graph")) for match in GRAPH_URL_RE.finditer(vector.group("windows")))


@validate_call
def window_state_for(graph_name: str) -> WindowState:
    """Whether *graph_name* has a Roam Desktop window open.

    Args:
        graph_name: A graph's canonical Roam name, matched case-insensitively — the same
            spelling that addresses it in a Local API path.

    Returns:
        ``OPEN`` when a window names the graph, ``CLOSED`` when the window state is
        legible and names no such window, and ``UNKNOWN`` when it cannot be read.
    """
    names: Final[tuple[str, ...] | None] = open_graph_names()
    if names is None:
        return WindowState.UNKNOWN
    folded: Final[str] = graph_name.strip().casefold()
    return WindowState.OPEN if any(name.casefold() == folded for name in names) else WindowState.CLOSED
