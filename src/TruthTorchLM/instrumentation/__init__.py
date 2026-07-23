"""Latency instrumentation for the pure black-box UQ benchmark (protocol §5).

Upstream TruthTorchLM times nothing; this package is the whole measurement layer. It is
**inert unless explicitly enabled**, so importing TruthTorchLM normally leaves the
library's behaviour and performance unchanged.

Typical use from the harness::

    from TruthTorchLM.instrumentation import capture, Stage

    with capture(method="DiscreteSemanticEntropy", n=5, generator="gpt-4o-mini") as rec:
        truth_value = method(model=..., ...)

    rec.marginal_ms()            # what the guardrail added beyond the user's answer
    rec.stage_ms(Stage.AUXILIARY_COMPUTE)   # the query-independent floor
"""

from .timing import (
    MARGINAL_STAGES,
    begin_stage,
    end_stage,
    Stage,
    StageEvent,
    TimingRecord,
    capture,
    current_record,
    disable,
    enable,
    is_enabled,
    record_metadata,
    stage,
)
from .stats import (
    DEFAULT_SLA_BUDGETS_MS,
    LatencySummary,
    overhead_matrix_row,
    sla_verdict,
    summarize,
)

__all__ = [
    "Stage",
    "StageEvent",
    "TimingRecord",
    "MARGINAL_STAGES",
    "stage",
    "begin_stage",
    "end_stage",
    "capture",
    "enable",
    "disable",
    "is_enabled",
    "current_record",
    "record_metadata",
    "LatencySummary",
    "summarize",
    "sla_verdict",
    "overhead_matrix_row",
    "DEFAULT_SLA_BUDGETS_MS",
    "sample_generations_api_concurrent",
]


def __getattr__(name):
    """Lazily expose the concurrent sampler, which needs litellm at import time.

    Keeps ``import TruthTorchLM.instrumentation`` usable (and testable) in a numpy-only
    environment while still offering the symbol at package level.
    """
    if name == "sample_generations_api_concurrent":
        from .concurrency import sample_generations_api_concurrent

        return sample_generations_api_concurrent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
