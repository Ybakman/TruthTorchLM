"""The guard against computing calibration metrics on uncalibrated scores.

Protocol §4B warns that verbalized and consistency scores are not probabilities. The
danger is not that the metric fails -- it is that it succeeds, returning a confident ECE
for a number that never meant anything. ``looks_uncalibrated`` detects the default
``SigmoidNormalizer(threshold=0, std=1.0)`` from the score arrays alone.
"""

import numpy as np

from TruthTorchLM.utils.calibration_metrics import looks_uncalibrated


def _default_sigmoid(x):
    """What TruthMethod.__init__'s dummy normalizer does to a raw truth value."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def test_detects_the_default_dummy_normalizer():
    raw = np.array([-3.2, -0.5, 0.0, 1.7, 4.4])
    assert looks_uncalibrated(raw, _default_sigmoid(raw)) is True


def test_a_fitted_normalizer_is_not_flagged():
    # An isotonic / min-max style map: monotone in the raw score but not the unit sigmoid.
    raw = np.array([-3.2, -0.5, 0.0, 1.7, 4.4])
    fitted = (raw - raw.min()) / (raw.max() - raw.min())
    assert looks_uncalibrated(raw, fitted) is False


def test_a_shifted_sigmoid_is_not_flagged():
    """SigmoidNormalizer.fit sets a real threshold and std -- that counts as calibrated."""
    raw = np.array([-3.2, -0.5, 0.0, 1.7, 4.4])
    calibrated = 1.0 / (1.0 + np.exp(-(raw - 0.8) / 2.1))
    assert looks_uncalibrated(raw, calibrated) is False


def test_handles_nan_and_inf_without_crashing():
    raw = np.array([np.nan, np.inf, 1.0])
    assert looks_uncalibrated(raw, _default_sigmoid(raw)) is True


def test_all_non_finite_is_not_a_verdict():
    raw = np.array([np.nan, np.nan])
    assert looks_uncalibrated(raw, np.array([np.nan, np.nan])) is False


def test_length_mismatch_is_not_a_verdict():
    assert looks_uncalibrated([1.0, 2.0], [0.5]) is False
