"""Calibration-error metrics, checked against cases with known closed-form answers.

These are deliberately hand-computable. A calibration metric that is subtly wrong still
returns a plausible number in [0, 1], so "it ran" proves nothing -- only known answers do.
"""

import numpy as np
import pytest

from TruthTorchLM.utils.calibration_metrics import (
    adaptive_calibration_error,
    brier_score,
    classwise_calibration_error,
    expected_calibration_error,
    kde_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)


def _perfectly_calibrated(n_per_level=200, seed=0):
    """Confidence p predicts correctness with probability exactly p."""
    rng = np.random.default_rng(seed)
    levels = np.linspace(0.05, 0.95, 10)
    conf = np.repeat(levels, n_per_level)
    labels = (rng.random(conf.size) < conf).astype(float)
    return conf, labels


class TestKnownAnswers:
    def test_perfectly_calibrated_has_near_zero_error(self):
        conf, labels = _perfectly_calibrated()
        assert expected_calibration_error(conf, labels) < 0.03
        assert adaptive_calibration_error(conf, labels) < 0.03
        assert maximum_calibration_error(conf, labels) < 0.08

    def test_confidently_wrong_is_maximally_miscalibrated(self):
        # Every response asserted at confidence 1.0; every one is wrong.
        conf = np.ones(100)
        labels = np.zeros(100)
        assert expected_calibration_error(conf, labels) == pytest.approx(1.0)
        assert maximum_calibration_error(conf, labels) == pytest.approx(1.0)
        assert brier_score(conf, labels) == pytest.approx(1.0)

    def test_confidently_right_is_perfect(self):
        conf = np.ones(100)
        labels = np.ones(100)
        assert expected_calibration_error(conf, labels) == pytest.approx(0.0)
        assert brier_score(conf, labels) == pytest.approx(0.0)

    def test_ece_equals_hand_computed_value(self):
        # Two bins, equal size. Bin A: conf 0.9, accuracy 0.5 -> gap 0.4.
        #                        Bin B: conf 0.1, accuracy 0.0 -> gap 0.1.
        # ECE = 0.5 * 0.4 + 0.5 * 0.1 = 0.25.  MCE = 0.4.
        conf = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert expected_calibration_error(conf, labels, n_bins=10) == pytest.approx(0.25)
        assert maximum_calibration_error(conf, labels, n_bins=10) == pytest.approx(0.4)

    def test_brier_matches_mean_squared_error(self):
        conf = np.array([0.8, 0.3, 0.6, 0.1])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        expected = np.mean((conf - labels) ** 2)
        assert brier_score(conf, labels) == pytest.approx(expected)

    def test_confidence_of_exactly_one_is_not_dropped(self):
        # The last bin must be closed, or every maximally-confident response --
        # the ones that matter most -- silently vanishes from the metric.
        conf = np.array([1.0, 1.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 0.0, 0.0])
        assert expected_calibration_error(conf, labels) == pytest.approx(1.0)
        rows = reliability_diagram_bins(conf, labels)
        assert sum(r["count"] for r in rows) == 4


class TestAdaptiveBinning:
    def test_ace_survives_the_high_confidence_pile_up(self):
        """The failure mode ACE exists to fix.

        Verbalized confidences cluster in 0.9-1.0. Equal-width binning drops nearly
        everything into one bin, so ECE reports a single coarse gap; equal-mass binning
        resolves the region and reports a larger, truer error.
        """
        rng = np.random.default_rng(1)
        conf = np.clip(rng.normal(0.95, 0.02, 2000), 0, 1)
        # Actual accuracy is only 0.6 -- badly overconfident.
        labels = (rng.random(conf.size) < 0.6).astype(float)
        ace = adaptive_calibration_error(conf, labels)
        assert ace > 0.3
        n_occupied = sum(1 for r in reliability_diagram_bins(conf, labels) if r["count"])
        n_occupied_adaptive = sum(
            1 for r in reliability_diagram_bins(conf, labels, adaptive=True) if r["count"]
        )
        assert n_occupied_adaptive > n_occupied

    def test_identical_confidences_do_not_crash(self):
        conf = np.full(50, 0.7)
        labels = np.concatenate([np.ones(35), np.zeros(15)])
        assert adaptive_calibration_error(conf, labels) == pytest.approx(0.0, abs=1e-9)


class TestKDEAndClasswise:
    def test_kde_agrees_with_ece_on_a_clean_case(self):
        conf, labels = _perfectly_calibrated(n_per_level=300, seed=7)
        assert kde_calibration_error(conf, labels) < 0.05

    def test_kde_flags_gross_overconfidence(self):
        conf = np.full(500, 0.95)
        labels = np.zeros(500)
        assert kde_calibration_error(conf, labels) == pytest.approx(0.95, abs=0.01)

    def test_classwise_weights_the_rare_error_class(self):
        """Global ECE is dominated by the majority class; class-wise is not.

        950 confidently-correct responses and 50 confidently-wrong ones: the errors are
        the minority, exactly as in health, and they are the ones that cost.
        """
        conf = np.concatenate([np.full(950, 0.95), np.full(50, 0.95)])
        labels = np.concatenate([np.ones(950), np.zeros(50)])
        global_ece = expected_calibration_error(conf, labels)
        cw = classwise_calibration_error(conf, labels)
        assert global_ece < 0.06
        assert cw > global_ece


class TestValidation:
    def test_rejects_raw_truth_values_outside_unit_interval(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            expected_calibration_error([-2.3, 0.5, 4.1], [1, 0, 1])

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            expected_calibration_error([0.5, np.nan], [1, 0])

    def test_rejects_the_minus_one_did_not_attempt_sentinel(self):
        with pytest.raises(ValueError, match="binary"):
            expected_calibration_error([0.5, 0.5], [1, -1])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error([0.5, 0.6], [1])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            expected_calibration_error([], [])
