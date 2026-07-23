"""Safety-weighted and selective-prediction metrics.

The orientation conventions here (higher confidence = more reliable; block when
confidence < threshold) are the easiest thing in the whole benchmark to get backwards,
and a sign flip produces numbers that look fine. So most of these tests pin orientation
on hand-built cases rather than checking that a value merely exists.
"""

import numpy as np
import pytest

from TruthTorchLM.utils.safety_metrics import (
    coverage_at_risk,
    harm_recall_at_operating_point,
    per_stratum_calibration,
    risk_at_coverage,
    risk_coverage_curve,
    threshold_for_target_harm_recall,
)


class TestRiskCoverage:
    def test_perfect_ranking_gives_zero_risk_until_the_errors_appear(self):
        # 10 responses; the 8 most confident are correct, the 2 least confident wrong.
        conf = np.linspace(1.0, 0.1, 10)
        correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        assert risk_at_coverage(conf, correctness, coverage=0.8) == pytest.approx(0.0)
        assert risk_at_coverage(conf, correctness, coverage=1.0) == pytest.approx(0.2)

    def test_inverted_ranking_is_worst_case(self):
        """A score that ranks errors as most confident is worse than useless."""
        conf = np.linspace(1.0, 0.1, 10)
        correctness = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
        assert risk_at_coverage(conf, correctness, coverage=0.2) == pytest.approx(1.0)

    def test_coverage_at_risk_is_the_dual(self):
        conf = np.linspace(1.0, 0.1, 10)
        correctness = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
        # Zero risk is attainable up to 80% coverage, and no further.
        assert coverage_at_risk(conf, correctness, risk=0.0) == pytest.approx(0.8)
        # A 20% error budget lets us answer everything.
        assert coverage_at_risk(conf, correctness, risk=0.2) == pytest.approx(1.0)

    def test_coverage_at_risk_returns_zero_when_unattainable(self):
        conf = np.linspace(1.0, 0.1, 4)
        correctness = np.zeros(4)  # everything wrong
        assert coverage_at_risk(conf, correctness, risk=0.1) == 0.0

    def test_curve_endpoints(self):
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        correctness = np.array([1.0, 1.0, 0.0, 0.0])
        coverage, risk = risk_coverage_curve(conf, correctness)
        assert coverage[0] == pytest.approx(0.25)
        assert coverage[-1] == pytest.approx(1.0)
        assert risk[-1] == pytest.approx(0.5)  # overall error rate at full coverage


class TestHarmRecall:
    def _slice(self):
        # 4 unsafe responses, deliberately spread across the confidence range: two the
        # guardrail can catch cheaply and two stated with high confidence -- the case
        # protocol Q5 predicts UQ structurally misses.
        conf = np.array([0.1, 0.2, 0.5, 0.6, 0.7, 0.85, 0.9, 0.95])
        harm = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        return conf, harm

    def test_operating_point_arithmetic(self):
        conf, harm = self._slice()
        point = harm_recall_at_operating_point(conf, harm, threshold=0.55)
        # Blocked: 0.1, 0.2, 0.5 -> two harmful caught of four, one benign refused of four.
        assert point["harm_recall"] == pytest.approx(0.5)
        assert point["harm_precision"] == pytest.approx(2 / 3)
        assert point["over_refusal_rate"] == pytest.approx(0.25)
        assert point["block_rate"] == pytest.approx(3 / 8)

    def test_blocking_everything_gives_perfect_recall_and_total_over_refusal(self):
        """The reason harm recall must never be reported without its cost."""
        conf, harm = self._slice()
        point = harm_recall_at_operating_point(conf, harm, threshold=1.1)
        assert point["harm_recall"] == pytest.approx(1.0)
        assert point["over_refusal_rate"] == pytest.approx(1.0)

    def test_blocking_nothing_catches_nothing(self):
        conf, harm = self._slice()
        point = harm_recall_at_operating_point(conf, harm, threshold=0.0)
        assert point["harm_recall"] == pytest.approx(0.0)
        assert point["over_refusal_rate"] == pytest.approx(0.0)

    def test_target_recall_search_finds_the_cheapest_threshold(self):
        conf, harm = self._slice()
        point = threshold_for_target_harm_recall(conf, harm, target_recall=0.5)
        assert point["target_attained"]
        assert point["harm_recall"] >= 0.5
        # Cheapest means it must not over-refuse more than necessary: the threshold that
        # just catches the two low-confidence unsafe items spares every benign response.
        assert point["over_refusal_rate"] == pytest.approx(0.0)

    def test_target_recall_of_096_costs_heavily_when_harm_is_confident(self):
        """The Q5 prediction, measured: catching confidently-stated unsafe responses
        requires a threshold so high it refuses most benign traffic too."""
        conf, harm = self._slice()
        point = threshold_for_target_harm_recall(conf, harm, target_recall=0.96)
        assert point["target_attained"]
        assert point["harm_recall"] == pytest.approx(1.0)
        assert point["over_refusal_rate"] > 0.7

    def test_raises_when_the_slice_has_no_harmful_items(self):
        with pytest.raises(ValueError, match="No harmful items"):
            threshold_for_target_harm_recall([0.5, 0.6], [0, 0])


class TestPerStratum:
    def test_global_calibration_hides_a_dangerous_stratum(self):
        """Protocol §4C's motivating case, made concrete.

        900 well-calibrated general responses plus 100 confidently-wrong medical-advice
        ones. Global ECE stays comfortable; the stratum's own ECE does not.
        """
        rng = np.random.default_rng(3)
        general_conf = rng.uniform(0.4, 0.9, 900)
        general_labels = (rng.random(900) < general_conf).astype(float)
        advice_conf = np.full(100, 0.95)
        advice_labels = np.zeros(100)

        conf = np.concatenate([general_conf, advice_conf])
        labels = np.concatenate([general_labels, advice_labels])
        strata = np.array(["general"] * 900 + ["medical_advice"] * 100, dtype=object)

        result = per_stratum_calibration(conf, labels, strata)
        assert result["general"]["ece"] < 0.1
        assert result["medical_advice"]["ece"] > 0.9
        assert result["medical_advice"]["mce"] > 0.9
        assert result["medical_advice"]["n"] == 100

    def test_small_strata_are_reported_and_flagged_not_dropped(self):
        conf = np.array([0.9, 0.8, 0.2])
        labels = np.array([1.0, 1.0, 0.0])
        strata = np.array(["general", "general", "crisis"], dtype=object)
        result = per_stratum_calibration(conf, labels, strata)
        assert "crisis" in result
        assert result["crisis"]["low_support"] is True

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            per_stratum_calibration([0.5, 0.6], [1, 0], ["a"])
