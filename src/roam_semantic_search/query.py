"""Hybrid retrieval over an index store: vector KNN and keyword BM25, RRF-fused.

Public symbols:

- :data:`RRF_K` — the reciprocal-rank-fusion damping constant.
- :class:`SearchHit` — one ranked search result.
- :func:`rrf_fused` — fuse rankings into one score per item (reciprocal rank fusion).
- :func:`search_store` — answer a natural-language query from a store.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.embed import DEFAULT_OLLAMA_URL, embed_query
from roam_semantic_search.store import (
    StoredRecord,
    keyword_ranked_uids,
    load_embedding_matrix,
    read_meta,
    records_by_uid,
)

RRF_K: Final[int] = 60
"""Reciprocal-rank-fusion damping constant (the conventional value)."""

_CANDIDATE_POOL_MIN: Final[int] = 50
_CANDIDATE_POOL_FACTOR: Final[int] = 4


class SearchHit(BaseModel):
    """One ranked search result.

    Attributes:
        uid: The hit's stable identifier (its Roam block/page uid).
        score: The fused retrieval score (higher is better).
        vector_rank: 1-based rank in the vector KNN ranking, when present.
        keyword_rank: 1-based rank in the keyword BM25 ranking, when present.
        page_title: Title of the page the hit belongs to.
        breadcrumb: The hit's context path.
        text: The hit's plain text.
        is_page: Whether the hit is a page rather than a block.
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


def _vector_ranked_uids(
    query_vector: Sequence[float], uids: Sequence[str], matrix: NDArray[np.float32], limit: int
) -> list[str]:
    """Cosine-similarity-ranked uids for a query vector, best first."""
    if matrix.shape[0] == 0:
        return []
    matrix_norms: Final[NDArray[np.float32]] = np.linalg.norm(matrix, axis=1).astype(np.float32)
    query_array: Final[NDArray[np.float32]] = np.asarray(query_vector, dtype=np.float32)
    query_norm: Final[float] = float(np.linalg.norm(query_array))
    similarities: Final[NDArray[np.float32]] = ((matrix @ query_array) / (matrix_norms * query_norm + 1e-12)).astype(
        np.float32
    )
    best_first: Final[NDArray[np.intp]] = np.argsort(-similarities)[:limit]
    return [uids[int(index)] for index in best_first]


@validate_call
def search_store(
    db_path: Path,
    query_text: str,
    k: int = 10,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[SearchHit]:
    """Answer a natural-language query from a store, hybrid-retrieved and RRF-fused.

    The query embeds with the store's own model; vector KNN and keyword BM25 rankings
    (each over a candidate pool larger than *k*) fuse by reciprocal rank fusion.

    Args:
        db_path: The store's database file.
        query_text: The natural-language query.
        k: Maximum hits returned.
        ollama_url: The local embedding server's base URL.

    Returns:
        The top-*k* hits, best fused score first.
    """
    pool: Final[int] = max(k * _CANDIDATE_POOL_FACTOR, _CANDIDATE_POOL_MIN)
    query_vector: Final[list[float]] = embed_query(
        query_text, model=read_meta(db_path).embed_model, base_url=ollama_url
    )
    uids, matrix = load_embedding_matrix(db_path)
    vector_ranking: Final[list[str]] = _vector_ranked_uids(query_vector, uids, matrix, pool)
    keyword_ranking: Final[list[str]] = keyword_ranked_uids(db_path, query_text, pool)
    fused: Final[dict[str, float]] = rrf_fused([vector_ranking, keyword_ranking])
    top_uids: Final[list[str]] = sorted(fused, key=lambda uid: fused[uid], reverse=True)[:k]

    records: Final[dict[str, StoredRecord]] = records_by_uid(db_path, top_uids)
    vector_rank_by_uid: Final[dict[str, int]] = {uid: rank for rank, uid in enumerate(vector_ranking, start=1)}
    keyword_rank_by_uid: Final[dict[str, int]] = {uid: rank for rank, uid in enumerate(keyword_ranking, start=1)}
    hits: Final[list[SearchHit]] = []
    for uid in top_uids:
        record: StoredRecord | None = records.get(uid)
        if record is None:
            continue
        hits.append(
            SearchHit(
                uid=uid,
                score=fused[uid],
                vector_rank=vector_rank_by_uid.get(uid),
                keyword_rank=keyword_rank_by_uid.get(uid),
                page_title=record.page_title,
                breadcrumb=record.breadcrumb,
                text=record.text,
                is_page=record.is_page,
            )
        )
    return hits
