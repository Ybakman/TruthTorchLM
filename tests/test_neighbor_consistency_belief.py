"""Neighbor-Consistency Belief (NCB) — the ported belief-score math.

NCB is pure numpy, so the whole scoring path is tested here without a model. The values
are pinned against the source formulas (P(y) x aggregate(p_i), the three aggregations, the
validity gate) computed by hand, so a porting slip fails loudly.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load():
    dotted = "TruthTorchLM.truth_methods.neighbor_consistency_belief"
    if dotted in sys.modules:
        return sys.modules[dotted]
    import types

    for pkg in ("TruthTorchLM", "TruthTorchLM.truth_methods"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(
        dotted, SRC / "TruthTorchLM/truth_methods/neighbor_consistency_belief.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


ncb = _load()


class TestOriginalQuestion:
    def test_p_y_is_the_majority_vote_fraction(self):
        dom, p_y = ncb.dominant_answer_probability(["paris", "paris", "paris", "lyon"])
        assert dom == "paris"
        assert p_y == pytest.approx(3 / 4)

    def test_no_samples_returns_none(self):
        assert ncb.dominant_answer_probability([]) == (None, None)

    def test_loose_match_uses_containment(self):
        assert ncb.is_dominant_correct("paris", "paris, france", "loose") is True
        assert ncb.is_dominant_correct("berlin", "paris", "loose") is False

    def test_strict_match_is_exact(self):
        assert ncb.is_dominant_correct("paris", "paris, france", "strict") is False
        assert ncb.is_dominant_correct("paris", "paris", "strict") is True


class TestNeighborAccuracy:
    def test_exact_case_insensitive_match_fraction(self):
        assert ncb.neighbor_accuracy(["Paris", "paris ", "LYON"], "paris") == pytest.approx(2 / 3)

    def test_no_responses_is_zero(self):
        assert ncb.neighbor_accuracy([], "paris") == 0.0


class TestAggregations:
    def test_geometric_mean_is_the_default(self):
        p_y, probs = 0.8, [0.5, 0.25]
        expected = p_y * math.sqrt(0.5 * 0.25)  # P(y) * geometric mean
        assert ncb.aggregate_neighbor_probs(probs, p_y, "geo_mean") == pytest.approx(expected)

    def test_arithmetic_mean(self):
        p_y, probs = 0.8, [0.5, 0.25]
        expected = p_y * ((0.5 + 0.25) / 2)
        assert ncb.aggregate_neighbor_probs(probs, p_y, "arith_mean") == pytest.approx(expected)

    def test_robust_geo_mean_drops_the_single_worst_neighbor(self):
        p_y, probs = 0.9, [0.9, 0.8, 0.01]  # the 0.01 is dropped
        expected = p_y * math.sqrt(0.9 * 0.8)
        assert ncb.aggregate_neighbor_probs(probs, p_y, "robust_geo_mean") == pytest.approx(expected)

    def test_no_neighbors_scores_zero(self):
        assert ncb.aggregate_neighbor_probs([], 0.9, "geo_mean") == 0.0

    def test_zero_neighbor_prob_does_not_nan_the_geo_mean(self):
        """EPSILON clipping: a neighbor at p_i=0 pulls the score toward 0 without log(0)."""
        score = ncb.aggregate_neighbor_probs([0.0, 0.9], 0.9, "geo_mean")
        assert score >= 0.0 and math.isfinite(score)
        assert score < 1e-4  # dominated by the clipped-near-zero neighbor

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="neighbor_agg"):
            ncb.aggregate_neighbor_probs([0.5], 0.5, "median")


class TestFullNCB:
    def test_illusion_of_confidence_self_consistent_but_fragile(self):
        """The paper's headline case: perfect original self-consistency (P(y)=1.0) but the
        belief collapses on neighbors -> low NCB, unlike plain self-consistency which is 1.0."""
        result = ncb.neighbor_consistency_belief(
            entities=["paris"] * 10,               # P(y) = 1.0
            golden_answer="paris",
            neighbor_responses=[["wrong"] * 10, ["nope"] * 10],  # neighbors all wrong -> p_i=0
            neighbor_correct_answers=["a", "b"],
        )
        assert result["valid"] is True
        assert result["p_y"] == pytest.approx(1.0)
        assert result["truth_value"] < 1e-4       # NCB collapses even though P(y)=1

    def test_robust_belief_scores_high(self):
        result = ncb.neighbor_consistency_belief(
            entities=["paris"] * 10,
            golden_answer="paris",
            neighbor_responses=[["a"] * 10, ["b"] * 10],   # neighbors all correct -> p_i=1
            neighbor_correct_answers=["a", "b"],
        )
        assert result["truth_value"] == pytest.approx(1.0)

    def test_wrong_dominant_answer_is_invalid_and_zero(self):
        """Gate: belief robustness is undefined when the model doesn't know the fact."""
        result = ncb.neighbor_consistency_belief(
            entities=["berlin"] * 10, golden_answer="paris",
            neighbor_responses=[["a"] * 10], neighbor_correct_answers=["a"],
        )
        assert result["valid"] is False
        assert result["reason"] == "wrong_answer"
        assert result["truth_value"] == 0.0

    def test_no_samples_is_invalid(self):
        result = ncb.neighbor_consistency_belief([], "paris", [], [])
        assert result["valid"] is False
        assert result["reason"] == "no_samples"

    def test_truth_value_mirrors_score(self):
        result = ncb.neighbor_consistency_belief(
            entities=["paris"] * 4, golden_answer="paris",
            neighbor_responses=[["a", "a", "b", "a"]], neighbor_correct_answers=["a"],
        )
        assert result["truth_value"] == result["score"]
