"""The Q1 deliverable: accuracy vs marginal wall-clock latency (protocol §5).

Best accuracy metric on the y-axis against **marginal** milliseconds on the x-axis (the
guardrail's added cost, not total), with the sample budget N swept along each method's
curve and **serial vs concurrent drawn as separate series**. That last split is the whole
point: it shows whether concurrency rescues the multi-sample family or leaves the N-times
penalty intact, and where a proxy or single-pass method sits relative to both.

x is log-scaled because the methods span roughly three orders of magnitude in latency (a
verbalized single pass vs an SC sweep at N=20 on a slow generator), and a linear axis
would collapse the cheap end into the y-axis. Whiskers are p95; the point is p50.
"""

__all__ = ["plot_frontier"]


def plot_frontier(
    series,
    accuracy_metric: str = "auroc",
    title: str = "Accuracy-latency frontier (marginal ms)",
    sla_budgets_ms=(500.0, 1000.0, 2000.0),
    out_path: str = None,
):
    """Render the frontier.

    ``series`` is a list of dicts, one per (method, execution mode)::

        {
          "label": "DiscreteSemanticEntropy (concurrent)",
          "execution": "concurrent",           # styles the line
          "points": [
             {"n": 1,  "accuracy": 0.71, "marginal_p50_ms": 30,  "marginal_p95_ms": 45},
             {"n": 5,  "accuracy": 0.79, "marginal_p50_ms": 120, "marginal_p95_ms": 180},
             ...
          ],
        }

    Serial series are drawn solid, concurrent dashed, so the two are distinguishable in
    grayscale (print) as well as colour. SLA budgets are drawn as vertical rules -- a
    method is admissible iff its curve has a point left of the budget line.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for s in series:
        pts = sorted(s["points"], key=lambda p: p["n"])
        x = [p["marginal_p50_ms"] for p in pts]
        y = [p["accuracy"] for p in pts]
        xerr = [max(0.0, p.get("marginal_p95_ms", p["marginal_p50_ms"]) - p["marginal_p50_ms"])
                for p in pts]
        linestyle = "--" if s.get("execution") == "concurrent" else "-"
        line = ax.plot(x, y, linestyle=linestyle, marker="o", label=s["label"])[0]
        ax.errorbar(x, y, xerr=[[0] * len(x), xerr], fmt="none",
                    ecolor=line.get_color(), alpha=0.4, capsize=3)
        for p in pts:  # annotate each point with its N
            ax.annotate(f"N={p['n']}", (p["marginal_p50_ms"], p["accuracy"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    for budget in sla_budgets_ms:
        ax.axvline(budget, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"{int(budget)}ms", (budget, ax.get_ylim()[0]),
                    rotation=90, fontsize=7, color="gray", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Marginal wall-clock latency (ms, p50; whiskers to p95)")
    ax.set_ylabel(accuracy_metric.upper())
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[frontier] wrote {out_path}")
    return fig, ax
