"""Text embedding through a locally hosted Ollama server.

Every call is refused unless the server URL points at the loopback interface — the
privacy guarantee that no text ever leaves the machine is enforced here, not merely
assumed.  The default model's task prefixes (``search_document:`` / ``search_query:``)
are applied by the callers' choice of prefix argument.

Public symbols:

- :data:`DEFAULT_EMBED_MODEL` — the default embedding model name.
- :data:`DEFAULT_OLLAMA_URL` — the default (loopback) Ollama base URL.
- :data:`DOCUMENT_PREFIX` / :data:`QUERY_PREFIX` — retrieval task prefixes.
- :data:`EMBED_BATCH_SIZE` — texts per embedding request.
- :func:`embed_texts` — embed a sequence of documents, batched.
- :func:`embed_query` — embed one query string.
"""

import logging
from collections.abc import Sequence
from typing import Final
from urllib.parse import urlparse

import requests
from pydantic import TypeAdapter, validate_call

from roam_semantic_search.json_narrowing import is_json_object

logger = logging.getLogger(__name__)

_EMBEDDINGS_ADAPTER: Final[TypeAdapter[list[list[float]]]] = TypeAdapter(list[list[float]])

DEFAULT_EMBED_MODEL: Final[str] = "nomic-embed-text"
"""Default embedding model name."""

DEFAULT_OLLAMA_URL: Final[str] = "http://127.0.0.1:11434"
"""Default Ollama base URL — the loopback interface."""

DOCUMENT_PREFIX: Final[str] = "search_document: "
"""Task prefix an indexed document embeds under (nomic-embed retrieval convention)."""

QUERY_PREFIX: Final[str] = "search_query: "
"""Task prefix a search query embeds under (nomic-embed retrieval convention)."""

EMBED_BATCH_SIZE: Final[int] = 32
"""Texts per embedding request."""

_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})

_REQUEST_TIMEOUT_SECONDS: Final[int] = 300


def _verified_loopback_url(base_url: str) -> str:
    """Return *base_url* unchanged, raising if its host is not a loopback interface.

    Args:
        base_url: The Ollama base URL to verify.

    Returns:
        The verified base URL.

    Raises:
        ValueError: If the URL's host is not a loopback interface.
    """
    host: Final[str | None] = urlparse(base_url).hostname
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(f"embedding server must be local (loopback), got host {host!r} in {base_url!r}")
    return base_url


def _embedded_batch(texts: Sequence[str], model: str, base_url: str) -> list[list[float]]:
    """Embed one batch of texts through the Ollama embed API.

    Args:
        texts: The (already prefixed) texts to embed.
        model: The embedding model name.
        base_url: The (already verified) Ollama base URL.

    Returns:
        One embedding vector per input text, in input order.

    Raises:
        requests.exceptions.HTTPError: If the server answers a non-200 status.
        TypeError: If the response is not a JSON object, or the embedding count disagrees
            with the input count.
        pydantic.ValidationError: If the embeddings entry is not a list of float vectors.
    """
    response: Final[requests.Response] = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": list(texts)},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body: Final[object] = response.json()
    if not is_json_object(body):
        raise TypeError(f"expected a JSON object from /api/embed, got {type(body).__name__}")
    embeddings: Final[list[list[float]]] = _EMBEDDINGS_ADAPTER.validate_python(body.get("embeddings"))
    if len(embeddings) != len(texts):
        raise TypeError(f"embed response carries {len(embeddings)} embeddings for {len(texts)} inputs")
    return embeddings


@validate_call
def embed_texts(
    texts: Sequence[str],
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    prefix: str = DOCUMENT_PREFIX,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a sequence of texts, batched, through a local Ollama server.

    Args:
        texts: The texts to embed.
        model: The embedding model name.
        base_url: The Ollama base URL; must point at a loopback interface.
        prefix: Task prefix prepended to every text before embedding.
        batch_size: Texts per request.

    Returns:
        One embedding vector per input text, in input order.

    Raises:
        ValueError: If *base_url* is not a loopback URL.
        requests.exceptions.HTTPError: If the server answers a non-200 status.
    """
    verified_url: Final[str] = _verified_loopback_url(base_url)
    prefixed: Final[list[str]] = [f"{prefix}{text}" for text in texts]
    vectors: Final[list[list[float]]] = []
    for start in range(0, len(prefixed), batch_size):
        batch: Sequence[str] = prefixed[start : start + batch_size]
        vectors.extend(_embedded_batch(batch, model, verified_url))
        logger.info("embedded %d/%d texts", min(start + batch_size, len(prefixed)), len(prefixed))
    return vectors


@validate_call
def embed_query(
    query_text: str,
    model: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> list[float]:
    """Embed one query string through a local Ollama server.

    Args:
        query_text: The query to embed.
        model: The embedding model name.
        base_url: The Ollama base URL; must point at a loopback interface.

    Returns:
        The query's embedding vector.

    Raises:
        ValueError: If *base_url* is not a loopback URL.
        requests.exceptions.HTTPError: If the server answers a non-200 status.
    """
    verified_url: Final[str] = _verified_loopback_url(base_url)
    return _embedded_batch([f"{QUERY_PREFIX}{query_text}"], model, verified_url)[0]
