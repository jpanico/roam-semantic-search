"""Hybrid retrieval over an index store: vector KNN and keyword BM25, RRF-fused.

Public symbols:

- :data:`RRF_K` — the reciprocal-rank-fusion damping constant.
- :class:`SearchHit` — one ranked search result.
- :func:`rrf_fused` — fuse rankings into one score per item (reciprocal rank fusion).
- :func:`collapse_key` — what a hit is deduplicated by: its record, or for an attachment chunk its attachment.
- :func:`search_store` — answer a natural-language query from a store.

A long PDF yields hundreds of chunk records, which would otherwise fill the results with one
document; hits therefore collapse to the best-ranked chunk of each attachment (PDF or image).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL, embed_query
from roam_semantic_search.normalize import RecordKind
from roam_semantic_search.store import (
    StoredRecord,
    keyword_ranked_ids,
    load_embedding_matrix,
    read_meta,
    records_by_id,
    upgrade_store,
)

RRF_K: Final[int] = 60
"""Reciprocal-rank-fusion damping constant (the conventional value)."""

_CANDIDATE_POOL_MIN: Final[int] = 200
_CANDIDATE_POOL_FACTOR: Final[int] = 10
_ATTACHMENT_RECORD_ID_SEPARATORS: Final[tuple[str, ...]] = ("/pdf/", "/img/")


class SearchHit(BaseModel):
    """One ranked search result.

    Attributes:
        uid: The Roam uid to cite as ``((uid))`` — the page or block itself, or for a PDF
            chunk the block that embeds the PDF.
        score: The fused retrieval score (higher is better).
        vector_rank: 1-based rank in the vector KNN ranking, when present.
        keyword_rank: 1-based rank in the keyword BM25 ranking, when present.
        page_title: Title of the page the hit belongs to.
        breadcrumb: The hit's context path (for a PDF chunk, ending in the PDF's name).
        text: The hit's plain text (for a PDF chunk, the matching passage).
        is_page: Whether the hit is a page rather than a block.
        kind: What the hit is: ``page``, ``block``, ``pdf`` (a passage of an attached PDF), or
            ``image`` (text read from an attached image — today an SVG's).
        attachment_name: For a ``pdf`` or ``image`` hit, the name the attachment was
            uploaded under, when known.
        attachment_page: For a ``pdf`` hit, the 1-based page the passage is on.
        pdf_name: ``attachment_name`` for a ``pdf`` hit only (kept for earlier consumers).
        pdf_page: ``attachment_page`` for a ``pdf`` hit only (kept for earlier consumers).
    """

    model_config = ConfigDict(frozen=True)

    uid: str
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    page_title: str
    breadcrumb: str
    text: str
    is_page: bool
    kind: RecordKind
    attachment_name: str | None = None
    attachment_page: int | None = None
    pdf_name: str | None = None
    pdf_page: int | None = None


@validate_call
def rrf_fused(rankings: Sequence[Sequence[str]], rrf_k: int = RRF_K) -> dict[str, float]:
    """Fuse rankings into one score per item, by reciprocal rank fusion.

    Each item scores the sum, over the rankings it appears in, of ``1 / (rrf_k + rank)``
    (ranks 1-based) — so appearing early in several rankings beats appearing early in one.

    Args:
        rankings: The rankings to fuse, each best-first.
        rrf_k: The damping constant.

    Returns:
        The fused score per item.
    """
    scores: Final[dict[str, float]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    return scores


@validate_call
def collapse_key(record_id: str) -> str:
    """What a hit is deduplicated by: its record id, or for an attachment chunk its attachment.

    Args:
        record_id: The hit's record id (``<uid>/pdf/<key>/<page>.<chunk>`` for a PDF chunk,
            ``<uid>/img/<key>/1.<chunk>`` for an image's text).

    Returns:
        The record id itself, or ``<uid>/pdf/<key>`` / ``<uid>/img/<key>`` for an
        attachment chunk.
    """
    if not any(separator in record_id for separator in _ATTACHMENT_RECORD_ID_SEPARATORS):
        return record_id
    return record_id.rsplit("/", 1)[0]


def _vector_ranked_ids(
    query_vector: Sequence[float], record_ids: Sequence[str], matrix: NDArray[np.float32], limit: int
) -> list[str]:
    """Cosine-similarity-ranked record ids for a query vector, best first."""
    if matrix.shape[0] == 0:
        return []
    matrix_norms: Final[NDArray[np.float32]] = np.linalg.norm(matrix, axis=1).astype(np.float32)
    query_array: Final[NDArray[np.float32]] = np.asarray(query_vector, dtype=np.float32)
    query_norm: Final[float] = float(np.linalg.norm(query_array))
    similarities: Final[NDArray[np.float32]] = ((matrix @ query_array) / (matrix_norms * query_norm + 1e-12)).astype(
        np.float32
    )
    best_first: Final[NDArray[np.intp]] = np.argsort(-similarities)[:limit]
    return [record_ids[int(index)] for index in best_first]


@validate_call
def search_store(
    db_path: Path,
    query_text: str,
    k: int = 10,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[SearchHit]:
    """Answer a natural-language query from a store, hybrid-retrieved and RRF-fused.

    The query embeds with the store's own model; vector KNN and keyword BM25 rankings
    (each over a candidate pool well beyond *k*) fuse by reciprocal rank fusion, and the
    fused ranking collapses to the best-ranked chunk of each PDF (:func:`collapse_key`)
    before the top *k* are taken.  A schema-v2 store is upgraded in place first.

    Args:
        db_path: The store's database file.
        query_text: The natural-language query.
        k: Maximum hits returned.
        ollama_url: The local embedding server's base URL.

    Returns:
        The top-*k* hits, best fused score first.
    """
    upgrade_store(db_path)
    pool: Final[int] = max(k * _CANDIDATE_POOL_FACTOR, _CANDIDATE_POOL_MIN)
    query_vector: Final[list[float]] = embed_query(
        query_text, model=read_meta(db_path).embed_model, base_url=ollama_url
    )
    record_ids, matrix = load_embedding_matrix(db_path)
    vector_ranking: Final[list[str]] = _vector_ranked_ids(query_vector, record_ids, matrix, pool)
    keyword_ranking: Final[list[str]] = keyword_ranked_ids(db_path, query_text, pool)
    fused: Final[dict[str, float]] = rrf_fused([vector_ranking, keyword_ranking])

    top_ids: Final[list[str]] = []
    seen_keys: Final[set[str]] = set()
    for record_id in sorted(fused, key=lambda candidate: fused[candidate], reverse=True):
        key: str = collapse_key(record_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        top_ids.append(record_id)
        if len(top_ids) == k:
            break

    records: Final[dict[str, StoredRecord]] = records_by_id(db_path, top_ids)
    vector_rank_by_id: Final[dict[str, int]] = {rid: rank for rank, rid in enumerate(vector_ranking, start=1)}
    keyword_rank_by_id: Final[dict[str, int]] = {rid: rank for rank, rid in enumerate(keyword_ranking, start=1)}
    hits: Final[list[SearchHit]] = []
    for record_id in top_ids:
        record: StoredRecord | None = records.get(record_id)
        if record is None:
            continue
        hits.append(
            SearchHit(
                uid=record.uid,
                score=fused[record_id],
                vector_rank=vector_rank_by_id.get(record_id),
                keyword_rank=keyword_rank_by_id.get(record_id),
                page_title=record.page_title,
                breadcrumb=record.breadcrumb,
                text=record.text,
                is_page=record.is_page,
                kind=record.kind,
                attachment_name=record.pdf_name,
                attachment_page=record.pdf_page if record.kind is RecordKind.PDF else None,
                pdf_name=record.pdf_name if record.kind is RecordKind.PDF else None,
                pdf_page=record.pdf_page if record.kind is RecordKind.PDF else None,
            )
        )
    return hits
