"""Stage D -- evaluate (protocol §4 metric suite, §5 frontier, §6 statistics).

Turns Stage-C scores + Stage-B labels into the reportable numbers. Three things the
protocol insists on are enforced here rather than left to discipline:

* **Per-dataset, never average-only** (§6). Averaging is what hid DisAAD's biomedical
  weakness; this module reports each dataset and only then a pool.
* **Multi-seed mean +/- std and bootstrap CIs on AUROC/AUPR** (§6). DisAAD's own tables
  swing +/-3-5 AUROC, so a single-run number is not a result. ``bootstrap_ci`` and
  ``aggregate_seeds`` make the spread explicit.
* **Both correctness criteria** (§3). Every metric is computed under each of BLEURT and
  LLM-judge, and the sensitivity between them is reported, not hidden by a silent choice.
"""

import numpy as np

__all__ = ["evaluate_method", "bootstrap_ci", "aggregate_seeds"]

DEFAULT_METRICS = ["auroc", "auprc", "auarc", "prr"]
DEFAULT_CALIBRATION_METRICS = ["ece", "ace", "mce", "brier"]


def bootstrap_ci(metric_fn, correctness, truth_values, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a discrimination metric.

    Resamples items with replacement. Returns (point, lo, hi). Needed because, per §6, a
    bare AUROC on one seed is inside the noise floor of methods like DisAAD.
    """
    rng = np.random.default_rng(seed)
    correctness = np.asarray(correctness)
    truth_values = np.asarray(truth_values)
    n = len(correctness)

    point = metric_fn(correctness, truth_values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = correctness[idx]
        if len(np.unique(c)) < 2:  # a resample with one class has no AUROC
            continue
        stats.append(metric_fn(c, truth_values[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def evaluate_method(
    method_scores_by_n,
    correctness,
    metrics=None,
    calibration_metrics=None,
    truth_method=None,
    strata=None,
    harm_labels=None,
    latency_summaries_by_n=None,
    require_calibrated=False,
):
    """All metrics for one method across its N-sweep, at one seed, one dataset, one criterion.

    ``method_scores_by_n`` is ``{n: MethodScores}`` from Stage C. ``correctness`` is the
    Stage-B label vector for the chosen criterion. Returns ``{n: {metric: value, ...}}``,
    with latency merged in when ``latency_summaries_by_n`` is supplied so the accuracy and
    the milliseconds for a given N live in one record -- which is what the frontier plot
    consumes.

    ``require_calibrated`` defaults False here: Stage D typically fits a normalizer just
    before this call, but exploratory runs may not, and a hard failure mid-sweep is worse
    than an uncalibrated-but-flagged calibration number. The library-level default
    (``metric_score``) stays strict.
    """
    # Imported here rather than at module top so the pure-numpy parts of Stage D
    # (bootstrap_ci, aggregate_seeds) stay importable without the full ML stack.
    from TruthTorchLM.utils.eval_utils import metric_score

    metrics = metrics or DEFAULT_METRICS
    calibration_metrics = calibration_metrics or DEFAULT_CALIBRATION_METRICS
    all_metrics = list(metrics) + list(calibration_metrics)

    out = {}
    for n, scores in method_scores_by_n.items():
        eval_dict = metric_score(
            all_metrics,
            correctness,
            scores["truth_values"],
            scores["normalized_truth_values"],
            truth_method=truth_method,
            strata=strata,
            harm_labels=harm_labels,
            require_calibrated=require_calibrated,
        )
        if latency_summaries_by_n and n in latency_summaries_by_n:
            eval_dict["latency"] = latency_summaries_by_n[n]
        out[n] = eval_dict
    return out


def aggregate_seeds(per_seed_evals):
    """Combine per-seed metric dicts into mean +/- std (protocol §6).

    ``per_seed_evals`` is a list of ``{n: {metric: value}}`` -- one per seed. Returns
    ``{n: {metric: {"mean":, "std":, "n_seeds":}}}``. Non-numeric fields (nested latency
    dicts, harm operating points) are passed through from the first seed unaggregated.
    """
    if not per_seed_evals:
        return {}

    ns = per_seed_evals[0].keys()
    aggregated = {}
    for n in ns:
        aggregated[n] = {}
        metric_names = per_seed_evals[0][n].keys()
        for metric in metric_names:
            values = [ev[n][metric] for ev in per_seed_evals if metric in ev[n]]
            numeric = [v for v in values if isinstance(v, (int, float)) and not _is_nan(v)]
            if numeric and len(numeric) == len(values):
                aggregated[n][metric] = {
                    "mean": float(np.mean(numeric)),
                    "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
                    "n_seeds": len(numeric),
                }
            else:
                aggregated[n][metric] = values[0]  # non-numeric: pass first seed through
    return aggregated


def _is_nan(v):
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False
