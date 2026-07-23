"""Q2 orchestration -- robustness under distribution shift (fit-on-A / test-on-B).

The prediction under test: proxy methods degrade *more sharply* OOD than non-proxy
consistency methods -- a steeper AUROC drop and, worse, a silent calibration collapse
(the scores stay in [0,1] and look fine while ceasing to track accuracy), because a
distilled proxy has learned the in-domain target distribution and is extrapolating, while
a consistency method only degrades in step with the target itself.

The machinery is deliberately thin: a truth method's normalizer is *fitted on domain A*
(``calibrate_truth_method`` already does exactly this) and the method is then *evaluated on
domain B*. What is new is the orchestration that pairs them and reports the drop, both in
discrimination (AUROC) and in calibration (ECE/MCE) -- the latter being where the "silent"
collapse hides, since discrimination alone would not reveal it.
"""

import numpy as np

__all__ = ["ood_degradation", "graded_shift_curve"]


def ood_degradation(in_domain_eval, ood_eval, metrics=("auroc", "prr", "ece", "mce")):
    """Compare a method's in-domain vs OOD metrics at matched N.

    Both inputs are ``{n: {metric: value}}`` (Stage D output for domain A and domain B).
    Returns ``{n: {metric: {"in_domain":, "ood":, "drop":}}}`` where ``drop`` is oriented
    so that **positive = worse OOD** for every metric: discrimination metrics (higher is
    better) drop as in-domain minus OOD; calibration errors (lower is better) drop as OOD
    minus in-domain. This uniform orientation is what lets the proxy-vs-consistency
    comparison be read off a single sign.
    """
    higher_is_better = {"auroc", "auprc", "auarc", "prr"}
    out = {}
    for n in in_domain_eval:
        if n not in ood_eval:
            continue
        out[n] = {}
        for metric in metrics:
            if metric not in in_domain_eval[n] or metric not in ood_eval[n]:
                continue
            id_val = _scalar(in_domain_eval[n][metric])
            ood_val = _scalar(ood_eval[n][metric])
            drop = (id_val - ood_val) if metric in higher_is_better else (ood_val - id_val)
            out[n][metric] = {"in_domain": id_val, "ood": ood_val, "drop": drop}
    return out


def graded_shift_curve(evals_by_shift_level, metric="auroc"):
    """A metric vs graded distribution-shift distance, for the degradation-curve figure.

    ``evals_by_shift_level`` maps an ordered shift label (e.g. MMLU subject distance, or
    "in_domain"/"near"/"far") to that slice's ``{n: {metric: value}}``. Returns a per-N
    series of (shift_level, value), the shape the plot consumes.
    """
    series = {}
    for shift_level, eval_by_n in evals_by_shift_level.items():
        for n, metrics in eval_by_n.items():
            if metric in metrics:
                series.setdefault(n, []).append((shift_level, _scalar(metrics[metric])))
    return series


def _scalar(value):
    """Unwrap a possibly-aggregated metric (``{"mean": ...}``) to a float."""
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)
