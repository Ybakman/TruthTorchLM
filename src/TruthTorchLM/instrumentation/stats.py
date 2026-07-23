"""Latency distributions and SLA verdicts (benchmark protocol §5).

The protocol is explicit that a mean is not an answer: "the tail is what a real-time
guardrail feels". So everything here is distribution-first -- p50/p95/p99 with the raw
samples retained -- and the headline deliverable is a **pass/fail against an absolute
budget**, not a relative ordering. For an inline guardrail the question is admissibility.

Warm-up runs are discarded rather than averaged in. The first call to a method pays for
lazy model loading, CUDA context creation, and HTTP connection setup; folding that into
p99 would report a number no steady-state user ever experiences.
"""

import numpy as np

__all__ = [
    "LatencySummary",
    "summarize",
    "sla_verdict",
    "overhead_matrix_row",
    "DEFAULT_SLA_BUDGETS_MS",
]

#: Guardrail budgets the protocol asks for a verdict against.
DEFAULT_SLA_BUDGETS_MS = (500.0, 1000.0, 2000.0)

#: Trials discarded before summarizing, per protocol §5's measurement hygiene.
DEFAULT_WARMUP = 5


class LatencySummary(dict):
    """A percentile summary that keeps its samples, so CIs remain computable downstream."""

    @property
    def p50(self) -> float:
        return self["p50_ms"]

    @property
    def p95(self) -> float:
        return self["p95_ms"]

    @property
    def p99(self) -> float:
        return self["p99_ms"]


def summarize(
    samples_ms,
    warmup: int = DEFAULT_WARMUP,
    label: str = "",
    keep_samples: bool = True,
) -> LatencySummary:
    """Percentile summary of a latency sample, after discarding ``warmup`` leading trials.

    Raises if fewer than 100 measured trials survive -- not to be pedantic, but because
    p99 over 20 samples is the 20th-largest value dressed up as a percentile, and the
    protocol asks for at least 100 trials per method. Set ``warmup=0`` and accept the
    warning field for exploratory runs.
    """
    arr = np.asarray(list(samples_ms), dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"No latency samples to summarize for {label or 'this method'}.")

    discarded = min(warmup, max(0, arr.size - 1))
    measured = arr[discarded:]

    summary = LatencySummary(
        {
            "label": label,
            "n_trials": int(measured.size),
            "n_warmup_discarded": int(discarded),
            "mean_ms": float(measured.mean()),
            "std_ms": float(measured.std(ddof=1)) if measured.size > 1 else 0.0,
            "min_ms": float(measured.min()),
            "p50_ms": float(np.percentile(measured, 50)),
            "p95_ms": float(np.percentile(measured, 95)),
            "p99_ms": float(np.percentile(measured, 99)),
            "max_ms": float(measured.max()),
            # Surfaced rather than raised, so an exploratory run still produces numbers
            # that are visibly marked as under-powered instead of silently trusted.
            "underpowered": bool(measured.size < 100),
        }
    )
    if keep_samples:
        summary["samples_ms"] = measured.tolist()
    return summary


def sla_verdict(summary: LatencySummary, budgets_ms=DEFAULT_SLA_BUDGETS_MS) -> dict:
    """Pass/fail of **marginal p95** against each fixed guardrail budget.

    p95 rather than p50 because a guardrail that is fast most of the time and stalls one
    request in twenty is not a guardrail that fits a budget.
    """
    p95 = summary["p95_ms"]
    return {
        "p95_ms": p95,
        "verdicts": {f"{int(b)}ms": bool(p95 <= b) for b in budgets_ms},
        "fits_any": bool(any(p95 <= b for b in budgets_ms)),
        "tightest_budget_met_ms": (
            float(min(b for b in budgets_ms if p95 <= b))
            if any(p95 <= b for b in budgets_ms)
            else None
        ),
    }


def overhead_matrix_row(
    generator: str,
    family: str,
    method: str,
    baseline_summary: LatencySummary,
    method_summary: LatencySummary,
    execution: str = "serial",
    n: int = 1,
    budgets_ms=DEFAULT_SLA_BUDGETS_MS,
) -> dict:
    """One cell of protocol §5's generator × UQ-family matrix.

    Reports both numbers the protocol insists on together, because each alone misleads:

    * **marginal ms** (``t - g``) answers "does it fit *this* deployment's budget";
    * **overhead ratio** (``(t - g) / g``) answers "what is the method's structural cost,
      independent of how fast the generator happens to be".

    The pairing is what exposes the real axis -- coupling. A proxy method's overhead is a
    roughly constant number of milliseconds, so its *ratio* shrinks as the generator
    slows. A consistency method's overhead is a roughly constant *ratio* (≈ N−1 serial),
    so its absolute cost explodes on a slow generator. Reporting one column would make
    those look like the same finding.
    """
    g_p50 = baseline_summary["p50_ms"]
    g_p95 = baseline_summary["p95_ms"]
    marginal_p50 = method_summary["p50_ms"] - g_p50
    marginal_p95 = method_summary["p95_ms"] - g_p95

    marginal_summary = LatencySummary({**method_summary, "p95_ms": marginal_p95})
    return {
        "generator": generator,
        "family": family,
        "method": method,
        "execution": execution,
        "n": n,
        "baseline_g_p50_ms": g_p50,
        "baseline_g_p95_ms": g_p95,
        "total_t_p50_ms": method_summary["p50_ms"],
        "total_t_p95_ms": method_summary["p95_ms"],
        "marginal_p50_ms": marginal_p50,
        "marginal_p95_ms": marginal_p95,
        "overhead_ratio_p50": (marginal_p50 / g_p50) if g_p50 > 0 else None,
        "sla": sla_verdict(marginal_summary, budgets_ms),
        "underpowered": method_summary["underpowered"] or baseline_summary["underpowered"],
    }
