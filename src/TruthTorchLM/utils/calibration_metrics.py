"""Calibration-error metrics for truth values (benchmark protocol §4B).

Upstream TruthTorchLM ships score->probability *normalizers*
(``TruthTorchLM.normalizers``) but no calibration-*error* metrics. This module adds
them.

Every function here takes ``confidences`` in [0, 1] -- i.e. *normalized* truth values,
the output of a fitted normalizer -- and binary ``labels`` where 1 = the generation was
correct. Raw truth values (an entropy, a cluster count, a cosine similarity) are **not**
probabilities and must not be passed in; see ``metric_score`` in
``TruthTorchLM.utils.eval_utils``, which enforces this.

Conventions follow the standard definitions:

* ECE   -- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
* ACE   -- Nixon et al., *Measuring Calibration in Deep Learning*, CVPRW 2019
           (equal-mass "adaptive" bins rather than equal-width).
* MCE   -- Naeini et al., AAAI 2015 (worst-bin gap).
* Brier -- Brier 1950, the binary (two-class) form: mean squared error of the
           probability against the 0/1 outcome.
"""

import numpy as np

__all__ = [
    "expected_calibration_error",
    "adaptive_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "kde_calibration_error",
    "classwise_calibration_error",
    "reliability_diagram_bins",
    "looks_uncalibrated",
    "UncalibratedTruthValuesError",
    "CALIBRATION_METRIC_NAMES",
]


class UncalibratedTruthValuesError(RuntimeError):
    """Raised when a calibration metric is requested on uncalibrated truth values.

    A method still carrying TruthTorchLM's default dummy normalizer --
    ``SigmoidNormalizer(threshold=0, std=1.0)``, set in ``TruthMethod.__init__`` -- has
    never been fitted against observed accuracy. Squashing a raw semantic entropy through
    that sigmoid produces a number in [0, 1] that means nothing. It would still yield an
    ECE, and that ECE would be meaningless -- which is the failure this error exists to
    prevent. See benchmark protocol §4B.
    """


def looks_uncalibrated(truth_values, normalized_truth_values) -> bool:
    """True if the normalized values are just ``sigmoid(raw)`` -- the default normalizer.

    Detects the dummy normalizer from the score arrays alone, so it also protects callers
    working from a cached Stage-C score table rather than a live ``TruthMethod``.
    """
    raw = np.asarray(truth_values, dtype=float).ravel()
    norm = np.asarray(normalized_truth_values, dtype=float).ravel()
    if raw.size == 0 or raw.size != norm.size:
        return False
    finite = np.isfinite(raw) & np.isfinite(norm)
    if not finite.any():
        return False
    with np.errstate(over="ignore"):
        implied = 1.0 / (1.0 + np.exp(-raw[finite]))
    return bool(np.allclose(implied, norm[finite], atol=1e-6))

# Names accepted by TruthTorchLM.utils.eval_utils.metric_score.
CALIBRATION_METRIC_NAMES = [
    "ece",
    "ace",
    "mce",
    "brier",
    "kde_ece",
    "classwise_ece",
]


def _validate(confidences, labels):
    """Coerce to arrays and reject inputs that would silently produce nonsense."""
    conf = np.asarray(confidences, dtype=float).ravel()
    lab = np.asarray(labels, dtype=float).ravel()

    if conf.shape != lab.shape:
        raise ValueError(
            f"confidences and labels must have the same length, got {conf.shape} and {lab.shape}."
        )
    if conf.size == 0:
        raise ValueError("Cannot compute a calibration metric on an empty input.")
    if not np.all(np.isfinite(conf)):
        raise ValueError(
            "confidences contain NaN or inf. Calibration metrics require finite "
            "probabilities in [0, 1] -- pass normalized truth values, not raw ones."
        )
    if conf.min() < 0.0 or conf.max() > 1.0:
        raise ValueError(
            f"confidences must lie in [0, 1], got range [{conf.min():.4g}, {conf.max():.4g}]. "
            "These look like raw truth values; fit a normalizer first "
            "(TruthTorchLM.calibrate_truth_method)."
        )
    unique = np.unique(lab)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            f"labels must be binary 0/1 (1 = correct), got values {unique[:5]}. "
            "TruthTorchLM uses -1 for 'model did not attempt'; filter those out first."
        )
    return conf, lab


def _equal_width_bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _equal_mass_bin_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges holding (approximately) equal numbers of samples.

    This is what makes ACE robust where ECE is not: verbalized confidences pile up at
    0.9-1.0, so equal-width bins put nearly every sample in one bin and the remaining
    bins contribute noise (or nothing) to the average.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(confidences, quantiles)
    edges[0], edges[-1] = 0.0, 1.0
    # Collapse duplicate edges (heavy ties) so we never emit an empty degenerate bin.
    return np.unique(edges)


def _binned_gaps(confidences: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    """Yield (weight, |accuracy - confidence|) per non-empty bin.

    Bins are half-open [lo, hi) except the last, which is closed, so a confidence of
    exactly 1.0 is counted rather than dropped.
    """
    n = confidences.size
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = labels[mask].mean()
        confidence = confidences[mask].mean()
        yield count / n, abs(accuracy - confidence)


def expected_calibration_error(confidences, labels, n_bins: int = 15) -> float:
    """ECE: sample-weighted mean |accuracy - confidence| over equal-width bins.

    The protocol's baseline calibration metric. Note its known weakness -- it is a
    global average, so a small, dangerously miscalibrated high-risk region is diluted
    by a large well-calibrated one. Report MCE and per-stratum ECE alongside it.
    """
    conf, lab = _validate(confidences, labels)
    edges = _equal_width_bin_edges(n_bins)
    return float(sum(w * gap for w, gap in _binned_gaps(conf, lab, edges)))


def adaptive_calibration_error(confidences, labels, n_bins: int = 15) -> float:
    """ACE: ECE over equal-mass (quantile) bins instead of equal-width bins."""
    conf, lab = _validate(confidences, labels)
    edges = _equal_mass_bin_edges(conf, n_bins)
    if len(edges) < 2:  # every confidence identical
        return float(abs(lab.mean() - conf.mean()))
    return float(sum(w * gap for w, gap in _binned_gaps(conf, lab, edges)))


def maximum_calibration_error(confidences, labels, n_bins: int = 15) -> float:
    """MCE: the worst |accuracy - confidence| over any non-empty bin.

    The safety-critical calibration number for medical boundaries: it reports the
    single region where confidence is most wrong, which is precisely what an average
    is designed to hide.
    """
    conf, lab = _validate(confidences, labels)
    edges = _equal_width_bin_edges(n_bins)
    gaps = [gap for _, gap in _binned_gaps(conf, lab, edges)]
    return float(max(gaps)) if gaps else 0.0


def brier_score(confidences, labels) -> float:
    """Brier score: mean (confidence - label)^2. Lower is better; 0 is perfect.

    Joint accuracy + calibration: it penalizes the confidently-wrong and the
    uselessly-safe alike, which is why the protocol keeps it as a headline number.
    """
    conf, lab = _validate(confidences, labels)
    return float(np.mean((conf - lab) ** 2))


def kde_calibration_error(confidences, labels, bandwidth: float = None) -> float:
    """Binning-free ECE via Nadaraya-Watson (Gaussian-kernel) smoothing.

    Estimates E|E[y | conf] - conf| without committing to a bin count, removing the
    binning-scheme sensitivity that makes ECE values hard to compare across papers.
    ``bandwidth`` defaults to Silverman's rule of thumb.
    """
    conf, lab = _validate(confidences, labels)
    n = conf.size
    if bandwidth is None:
        std = conf.std(ddof=1) if n > 1 else 0.0
        if std <= 0:
            return float(abs(lab.mean() - conf.mean()))
        bandwidth = 1.06 * std * n ** (-1 / 5)
    bandwidth = max(float(bandwidth), 1e-6)

    # weights[i, j] = K((conf_i - conf_j) / h); the Gaussian normalizer cancels in the ratio.
    diff = (conf[:, None] - conf[None, :]) / bandwidth
    weights = np.exp(-0.5 * diff**2)
    smoothed_accuracy = (weights @ lab) / weights.sum(axis=1)
    return float(np.mean(np.abs(smoothed_accuracy - conf)))


def classwise_calibration_error(confidences, labels, n_bins: int = 15) -> float:
    """Class-wise ECE: the mean of ECE computed separately per outcome class.

    In health the error class is often the minority, so a global ECE is dominated by
    the correct class. Averaging the two per-class ECEs weights them equally, which
    surfaces miscalibration on the rare-but-costly side.

    For the incorrect class the relevant probability is 1 - confidence (the model's
    implied probability of that class).
    """
    conf, lab = _validate(confidences, labels)
    per_class = []
    for cls in (1.0, 0.0):
        mask = lab == cls
        if not mask.any():
            continue
        cls_conf = conf[mask] if cls == 1.0 else 1.0 - conf[mask]
        cls_lab = np.ones(int(mask.sum()))
        edges = _equal_width_bin_edges(n_bins)
        per_class.append(sum(w * gap for w, gap in _binned_gaps(cls_conf, cls_lab, edges)))
    return float(np.mean(per_class)) if per_class else 0.0


def reliability_diagram_bins(confidences, labels, n_bins: int = 15, adaptive: bool = False):
    """Per-bin (lo, hi, count, mean_confidence, accuracy) rows for a reliability plot.

    Returned rather than plotted so Stage D owns the figure styling.
    """
    conf, lab = _validate(confidences, labels)
    edges = _equal_mass_bin_edges(conf, n_bins) if adaptive else _equal_width_bin_edges(n_bins)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "mean_confidence": float(conf[mask].mean()) if count else None,
                "accuracy": float(lab[mask].mean()) if count else None,
            }
        )
    return rows
