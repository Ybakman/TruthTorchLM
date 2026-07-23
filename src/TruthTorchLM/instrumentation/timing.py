"""Wall-clock stage timing for UQ methods (benchmark protocol §5).

Upstream TruthTorchLM scores truth values but times nothing, so the entire latency layer
is new. The protocol's central claim is that "single-pass < multi-sample" is the *coarse*
story and hides the two things that actually decide deployability: whether a method fits
an absolute real-time budget, and whether concurrency collapses or leaves the N× penalty
intact. Both require measured milliseconds.

Design constraints this module is built around:

**Three stages, timed separately.** Total per-estimate latency decomposes into

    target generation  +  extra generations  +  auxiliary compute

Target generation is the answer the user gets anyway, so it is measured but *excluded*
from marginal latency. Extra generations (multi-sample draws, chain-of-interaction turns)
are the dominant marginal term and scale with the generator. Auxiliary compute (NLI
clustering, embedding, proxy forward pass) is the query-independent floor. Collapsing
these into one number would make the generator×family matrix in §5 unreadable.

**Off by default, and genuinely off.** When disabled, ``stage()`` returns a shared null
context manager: no clock reads, no allocation, no dict writes. Upstream behaviour has to
be bit-identical when the benchmark is not running, or the fork stops being a safe base
to merge upstream changes into.

**Context-local, not global.** Records live in a :class:`contextvars.ContextVar`, so
concurrent API sampling (``instrumentation.concurrency``) and threaded harness runs
accumulate into the right record instead of racing on one.
"""

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "Stage",
    "StageEvent",
    "TimingRecord",
    "stage",
    "begin_stage",
    "end_stage",
    "capture",
    "enable",
    "disable",
    "is_enabled",
    "current_record",
    "record_metadata",
]


class Stage(str, Enum):
    """The stage decomposition of protocol §5."""

    #: Producing the answer the user receives. Measured, but excluded from marginal cost.
    TARGET_GENERATION = "target_generation"
    #: Extra draws from the target: multi-sample consistency, chain-of-interaction turns.
    EXTRA_GENERATION = "extra_generation"
    #: Everything that is not a target call: NLI clustering, embeddings, proxy forward pass.
    AUXILIARY_COMPUTE = "auxiliary_compute"


#: Stages that count toward marginal latency -- i.e. everything the guardrail adds
#: beyond producing the user's answer. This is the number the SLA is written against.
MARGINAL_STAGES = (Stage.EXTRA_GENERATION, Stage.AUXILIARY_COMPUTE)


@dataclass
class StageEvent:
    """One timed span."""

    stage: Stage
    label: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TimingRecord:
    """All stage events for a single (item, method) estimate."""

    events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, event: StageEvent) -> None:
        self.events.append(event)

    # -- aggregate views -------------------------------------------------

    def total_ms(self) -> float:
        return sum(e.duration_ms for e in self.events)

    def stage_ms(self, stage: Stage) -> float:
        return sum(e.duration_ms for e in self.events if e.stage is stage)

    def marginal_ms(self) -> float:
        """Time the guardrail adds beyond producing the user's answer (§5)."""
        return sum(e.duration_ms for e in self.events if e.stage in MARGINAL_STAGES)

    def generation_baseline_ms(self) -> float:
        """``g`` -- generator-only time, the denominator of the overhead ratio."""
        return self.stage_ms(Stage.TARGET_GENERATION)

    def overhead_ratio(self) -> Optional[float]:
        """``(t - g) / g``: the method's structural cost, independent of generator speed.

        Returns None when no target generation was timed in this record -- which is the
        normal case in the cached harness, where Stage A produced the answer in an
        earlier pass. Stage D pairs the marginal number with the separately-measured
        baseline for that generator rather than inventing a ratio here.
        """
        baseline = self.generation_baseline_ms()
        if baseline <= 0:
            return None
        return self.marginal_ms() / baseline

    def label_ms(self, label: str) -> float:
        return sum(e.duration_ms for e in self.events if e.label == label)

    def as_dict(self) -> dict:
        return {
            "total_ms": self.total_ms(),
            "target_generation_ms": self.stage_ms(Stage.TARGET_GENERATION),
            "extra_generation_ms": self.stage_ms(Stage.EXTRA_GENERATION),
            "auxiliary_compute_ms": self.stage_ms(Stage.AUXILIARY_COMPUTE),
            "marginal_ms": self.marginal_ms(),
            "overhead_ratio": self.overhead_ratio(),
            "metadata": dict(self.metadata),
            "events": [
                {
                    "stage": e.stage.value,
                    "label": e.label,
                    "duration_ms": e.duration_ms,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
        }


# --------------------------------------------------------------------------
# Enable / disable
# --------------------------------------------------------------------------

_ENABLED = False
_CURRENT: contextvars.ContextVar = contextvars.ContextVar(
    "truthtorchlm_timing_record", default=None
)


def enable() -> None:
    """Turn instrumentation on process-wide. ``capture()`` does this for you."""
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def current_record() -> Optional[TimingRecord]:
    """The record spans are currently accumulating into, or None."""
    return _CURRENT.get()


@contextmanager
def _null_context():
    yield None


_NULL = _null_context


@contextmanager
def stage(stage: Stage, label: str = "", **metadata):
    """Time a span and attribute it to ``stage`` on the active record.

    A no-op -- no clock read at all -- when instrumentation is disabled or no record is
    active, so this is safe to leave in the hot path of the library.

    The span is recorded even if the body raises, because a method that times out or
    errors after 40 seconds is a latency fact worth keeping, not one to discard.
    """
    if not _ENABLED:
        yield None
        return
    record = _CURRENT.get()
    if record is None:
        yield None
        return

    start = time.perf_counter_ns()
    event = StageEvent(stage=stage, label=label or stage.value, duration_ms=0.0,
                       metadata=dict(metadata))
    try:
        yield event
    finally:
        event.duration_ms = (time.perf_counter_ns() - start) / 1e6
        record.add(event)


def begin_stage(stage: Stage, label: str = "", **metadata):
    """Open a timed span explicitly; close it with :func:`end_stage`.

    The ``with stage(...)`` form is preferred. This pair exists for the seams inside
    upstream ``generation.py``, where wrapping a long loop in a ``with`` block would
    re-indent code we want to keep byte-similar to upstream so future merges stay clean.

    Returns an opaque handle, or None when instrumentation is off.
    """
    if not _ENABLED:
        return None
    record = _CURRENT.get()
    if record is None:
        return None
    event = StageEvent(stage=stage, label=label or stage.value, duration_ms=0.0,
                       metadata=dict(metadata))
    return (record, event, time.perf_counter_ns())


def end_stage(span) -> None:
    """Close a span opened by :func:`begin_stage`. Accepts None and does nothing."""
    if span is None:
        return
    record, event, start = span
    event.duration_ms = (time.perf_counter_ns() - start) / 1e6
    record.add(event)


def record_metadata(**metadata) -> None:
    """Attach measurement-hygiene context (token counts, N, endpoint) to the active record."""
    record = _CURRENT.get()
    if record is not None:
        record.metadata.update(metadata)


@contextmanager
def capture(**metadata):
    """Collect all stage events raised inside the block into a fresh :class:`TimingRecord`.

    Enables instrumentation for the duration and restores the previous state after, so a
    harness can time one call without leaving the library instrumented::

        with capture(method="DiscreteSemanticEntropy", n=5) as rec:
            truth_value = method(...)
        rec.marginal_ms()
    """
    global _ENABLED
    was_enabled = _ENABLED
    _ENABLED = True
    record = TimingRecord(metadata=dict(metadata))
    token = _CURRENT.set(record)
    try:
        yield record
    finally:
        _CURRENT.reset(token)
        _ENABLED = was_enabled
