"""Phase 0 spike: full-graph pull against a live Roam graph via the Local API.

Measures the design plan's unverified assumptions (docs/design-plan.md, Phases):

1. Full-graph pull — every entity carrying a ``:block/uid``, pulled with guffin's
   aliased wildcard pattern: entity count, payload size, wall time.
2. Uid-only sweep (the deletion-detection query): wall time.
3. Edit-time candidate query (the incremental-refresh selector, exercising the
   ``[(> ?t ?since)]`` predicate through the Local API): candidate count, wall time.

Run with guffin's venv (guffin installed editable):

    /Users/jpanico/Documents/github/guffin/.venv/bin/python spike/phase0_full_graph_pull.py

Requires Roam Desktop running and the GUFFIN_ROAM_* env vars set.
"""

import json
import os
import time
from typing import Final

from guffin.roam.local_api import ApiEndpoint, Request, Response, invoke_action

FULL_PULL_QUERY: Final[str] = (
    '[:find (pull ?e [* [:block/view-type :as "block-view-type"]'
    ' [:children/view-type :as "children-view-type"]])'
    " :where [?e :block/uid]]"
)

UID_SWEEP_QUERY: Final[str] = "[:find ?uid :where [?e :block/uid ?uid]]"

EDIT_TIME_CANDIDATES_QUERY: Final[str] = (
    "[:find (pull ?e [:block/uid])" " :in $ ?since" " :where [?e :block/uid] [?e :edit/time ?t] [(> ?t ?since)]]"
)

SEVEN_DAYS_MS: Final[int] = 7 * 24 * 60 * 60 * 1000


def timed_query(api_endpoint: ApiEndpoint, query: str, args: list[object]) -> tuple[object, float]:
    """Run one data.q action and return (result, wall_seconds)."""
    payload: Final[Request.Payload] = Request.Payload(action="data.q", args=[query, *args])
    started: Final[float] = time.perf_counter()
    response: Final[Response.Payload] = invoke_action(payload, api_endpoint)
    elapsed: Final[float] = time.perf_counter() - started
    return response.result, elapsed


def main() -> None:
    """Run the three spike queries and print the measurements."""
    api_endpoint: Final[ApiEndpoint] = ApiEndpoint.from_parts(
        local_api_port=int(os.environ["GUFFIN_ROAM_LOCAL_API_PORT"]),
        graph_name=os.environ["GUFFIN_ROAM_GRAPH_NAME"],
        bearer_token=os.environ["GUFFIN_ROAM_API_TOKEN"],
    )
    print(f"graph: {api_endpoint.url}")

    # 1. Full-graph pull.
    full_result, full_seconds = timed_query(api_endpoint, FULL_PULL_QUERY, [])
    assert isinstance(full_result, list)
    rows: Final[list[dict[str, object]]] = [row[0] for row in full_result]
    payload_bytes: Final[int] = len(json.dumps(full_result).encode())

    pages: Final[int] = sum(1 for row in rows if "title" in row)
    blocks: Final[int] = sum(1 for row in rows if "string" in row)
    other: Final[int] = len(rows) - pages - blocks
    with_edit_time: Final[int] = sum(1 for row in rows if "edit-time" in row)
    empty_strings: Final[int] = sum(1 for row in rows if row.get("string", None) == "")
    text_chars: Final[int] = sum(
        len(value) for row in rows for value in (row.get("string"), row.get("title")) if isinstance(value, str)
    )

    print("\n=== 1. full-graph pull ===")
    print(f"wall time:        {full_seconds:.2f}s")
    print(f"payload (approx): {payload_bytes / 1_000_000:.1f} MB (re-serialized JSON)")
    print(f"entities:         {len(rows)} ({pages} pages, {blocks} blocks, {other} other)")
    print(f"with edit-time:   {with_edit_time}")
    print(f"empty strings:    {empty_strings}")
    print(f"text content:     {text_chars / 1000:.0f}k chars")

    # 2. Uid-only sweep (deletion detection).
    sweep_result, sweep_seconds = timed_query(api_endpoint, UID_SWEEP_QUERY, [])
    assert isinstance(sweep_result, list)
    print("\n=== 2. uid-only sweep ===")
    print(f"wall time: {sweep_seconds:.2f}s")
    print(f"uids:      {len(sweep_result)}")

    # 3. Edit-time candidates (incremental-refresh selector).
    since_ms: Final[int] = int(time.time() * 1000) - SEVEN_DAYS_MS
    candidates_result, candidates_seconds = timed_query(api_endpoint, EDIT_TIME_CANDIDATES_QUERY, [since_ms])
    assert isinstance(candidates_result, list)
    print("\n=== 3. edit-time candidates (last 7 days) ===")
    print(f"wall time:  {candidates_seconds:.2f}s")
    print(f"candidates: {len(candidates_result)}")


if __name__ == "__main__":
    main()
