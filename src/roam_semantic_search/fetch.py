"""Full-graph fetch from the Roam Local API.

Public symbols:

- :data:`FULL_PULL_QUERY` — Datalog query pulling every entity carrying a ``:block/uid``.
- :func:`fetch_graph` — run the full-graph pull and return the pull-block rows.
"""

import logging
from typing import Final

from guffin.roam.local_api import ApiEndpoint, Request, Response, invoke_action
from pydantic import validate_call

from roam_semantic_search.json_narrowing import is_json_list, is_json_object

logger = logging.getLogger(__name__)

FULL_PULL_QUERY: Final[str] = (
    '[:find (pull ?e [* [:block/view-type :as "block-view-type"]'
    ' [:children/view-type :as "children-view-type"]])'
    " :where [?e :block/uid]]"
)
"""Datalog query pulling every entity carrying a ``:block/uid`` — pages and blocks alike.

Uses the aliased wildcard pull pattern so the two ``view-type`` attributes cannot collide on one
namespace-stripped wire key (the ``:block/view-type`` / ``:children/view-type`` collision).
"""


@validate_call
def fetch_graph(api_endpoint: ApiEndpoint) -> list[dict[str, object]]:
    """Fetch every entity of the graph as one pull-block row per entity.

    Runs :data:`FULL_PULL_QUERY` through the Local API and unwraps the single-entity rows of the
    Datalog result.

    Args:
        api_endpoint: The Local API endpoint (URL + bearer token) for the target graph.

    Returns:
        One pull-block ``dict`` per entity, with namespace-stripped wire keys (``uid``, ``string``,
        ``title``, ``parents``, ...).

    Raises:
        requests.exceptions.ConnectionError: If the Local API is unreachable.
        requests.exceptions.HTTPError: If the Local API returns a non-200 status.
        TypeError: If the query result is not the expected rows-of-pull-blocks shape.
    """
    payload: Final[Request.Payload] = Request.Payload(action="data.q", args=[FULL_PULL_QUERY])
    response: Final[Response.Payload] = invoke_action(payload, api_endpoint)
    result: Final[object] = response.result
    if not is_json_list(result):
        raise TypeError(f"expected a list result from data.q, got {type(result).__name__}")
    rows: Final[list[dict[str, object]]] = []
    for row in result:
        if not is_json_list(row) or not row or not is_json_object(row[0]):
            raise TypeError("expected each data.q row to be a one-entity pull-block list")
        rows.append(row[0])
    logger.info("fetched %d entities from %s", len(rows), api_endpoint.url)
    return rows
