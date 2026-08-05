"""Tests for roam_semantic_search.query."""

from typing import Final

from roam_semantic_search.query import rrf_fused


class TestRrfFused:
    def test_item_in_both_rankings_beats_single_ranking_leader(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a", "b"], ["b", "c"]])
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_rank_order_within_one_ranking(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a", "b", "c"]])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_scores_are_reciprocal_rank_sums(self) -> None:
        scores: Final[dict[str, float]] = rrf_fused([["a"], ["a"]], rrf_k=60)
        assert scores["a"] == 2.0 / 61.0

    def test_empty_rankings_fuse_to_nothing(self) -> None:
        assert rrf_fused([]) == {}
        assert rrf_fused([[], []]) == {}
