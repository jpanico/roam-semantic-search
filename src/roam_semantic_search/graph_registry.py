"""Roam's own record of the graphs connected on this machine, and the endpoints they reach.

Roam Desktop keeps its Local API connection details in two files, which together identify
every connected graph:

- ``~/.roam-local-api.json`` — ``{"port": 3333}``.  **One port serves every graph**; the
  graph is named by the request path, not by a port of its own.
- ``~/.roam-tools.json`` — a ``graphs`` array of ``{name, nickname, token, type, ...}``
  entries, written when a graph is connected.

A graph is therefore addressable by the short nickname it was connected under, rather than
by carrying its canonical name, port, and bearer token around separately.

Public symbols:

- :data:`LOCAL_API_CONFIG_PATH` — where Roam records the shared Local API port.
- :data:`GRAPH_REGISTRY_PATH` — where Roam records the connected graphs.
- :class:`GraphType` — whether a graph's storage is cloud-hosted or local-only.
- :class:`RegisteredGraph` — one connected graph's identity and credential.
- :func:`local_api_port` — the shared Local API port.
- :func:`registered_graphs` — every connected graph, in registry order.
- :func:`resolve_graph` — the graph a nickname or canonical name selects.
- :func:`api_endpoint_for` — the Local API endpoint that reaches a graph.
"""

import enum
import json
from pathlib import Path
from typing import Final

from guffin.roam.local_api import ApiEndpoint
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validate_call

LOCAL_API_CONFIG_PATH: Final[Path] = Path.home() / ".roam-local-api.json"
GRAPH_REGISTRY_PATH: Final[Path] = Path.home() / ".roam-tools.json"


class GraphType(enum.StrEnum):
    """Where a graph's storage lives, as the registry records it.

    This says nothing about which API reaches the graph — every connected graph is
    reached through the Local API — only about where its data is held.
    """

    HOSTED = "hosted"
    OFFLINE = "offline"


class RegisteredGraph(BaseModel):
    """One graph as Roam's registry records it.

    Attributes:
        name: The canonical graph name, which is also its Local API path segment.
        nickname: The short label the graph was connected under.
        token: The Local API bearer token minted for this graph.
        graph_type: Whether the graph's storage is cloud-hosted or local-only.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    nickname: str = ""
    token: str = ""
    graph_type: GraphType = Field(default=GraphType.HOSTED, alias="type")

    @property
    def selectors(self) -> frozenset[str]:
        """The case-folded strings that select this graph."""
        return frozenset(part.lower() for part in (self.nickname, self.name) if part)


class _Registry(BaseModel):
    """The registry file as a whole, so its parse is typed rather than narrowed by hand.

    Unrecognized keys are ignored: Roam owns this file and may add fields at any time,
    and an unfamiliar one is no reason to stop resolving graphs.

    Attributes:
        graphs: The connected graphs, in the order the file lists them.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    graphs: tuple[RegisteredGraph, ...] = ()


@validate_call
def local_api_port() -> int:
    """The Local API port Roam serves every graph on.

    Returns:
        The port recorded in :data:`LOCAL_API_CONFIG_PATH`.

    Raises:
        RuntimeError: If the file is missing or records no usable port.
    """
    if not LOCAL_API_CONFIG_PATH.is_file():
        raise RuntimeError(f"no Local API config at {LOCAL_API_CONFIG_PATH} — is Roam Desktop installed?")
    try:
        return int(json.loads(LOCAL_API_CONFIG_PATH.read_text())["port"])
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{LOCAL_API_CONFIG_PATH} records no usable port") from exc


@validate_call
def registered_graphs() -> tuple[RegisteredGraph, ...]:
    """Every graph connected on this machine, in registry order.

    Returns:
        The registry's entries; empty when no graph has been connected, or when the file
        is absent or unreadable — an unusable registry is indistinguishable from an empty
        one to every caller, and neither is this module's error to raise.
    """
    if not GRAPH_REGISTRY_PATH.is_file():
        return ()
    try:
        return _Registry.model_validate_json(GRAPH_REGISTRY_PATH.read_text()).graphs
    except OSError, ValidationError:
        return ()


@validate_call
def resolve_graph(selector: str) -> RegisteredGraph:
    """Return the graph that *selector* names.

    Args:
        selector: A graph's nickname or its canonical name, matched case-insensitively.

    Returns:
        The matching registry entry.

    Raises:
        RuntimeError: If no connected graph answers to *selector*.
    """
    folded: Final[str] = selector.strip().lower()
    graphs: Final[tuple[RegisteredGraph, ...]] = registered_graphs()
    for graph in graphs:
        if folded in graph.selectors:
            return graph
    known: Final[str] = ", ".join(sorted(graph.nickname or graph.name for graph in graphs)) or "(none connected)"
    raise RuntimeError(f"unknown graph {selector!r}; connected graphs: {known}")


@validate_call
def api_endpoint_for(graph: RegisteredGraph) -> ApiEndpoint:
    """Return the Local API endpoint that reaches *graph*.

    Args:
        graph: The graph to build an endpoint for.

    Returns:
        The endpoint, carrying the shared port and the graph's own token.

    Raises:
        RuntimeError: If the graph is offline-typed, whose Local API path the
            ``/api/<name>`` form does not serve, or if it carries no token.
    """
    if graph.graph_type is GraphType.OFFLINE:
        raise RuntimeError(
            f"graph {graph.nickname or graph.name!r} is an offline graph; the /api/<name> Local API "
            f"path serves hosted graphs only"
        )
    if not graph.token:
        raise RuntimeError(f"graph {graph.nickname or graph.name!r} has no Local API token in the registry")
    return ApiEndpoint.from_parts(local_api_port=local_api_port(), graph_name=graph.name, bearer_token=graph.token)
