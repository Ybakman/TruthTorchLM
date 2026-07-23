"""hc_benchmark -- the Stage A-D orchestration for the pure black-box UQ benchmark.

This package is the *harness*, not part of the TruthTorchLM library. It sits on top of
the library and wires together the pieces the fork added -- latency instrumentation,
calibration/safety metrics, dataset loaders -- into the four-stage pipeline of the
benchmark protocol (`docs/benchmark_protocol.md`, §7), which mirrors the DisAAD repo
skeleton so the two are directly comparable:

    Stage A  generate   primary answer + N_max samples, cached to Parquet
    Stage B  label      correctness via BLEURT *and* LLM-judge, both persisted
    Stage C  score      each UQ method reads the cache, sweeping N in {1,3,5,10,20},
                        logging one timing record per (method, N, item)
    Stage D  evaluate   the §4 metric suite + §5 frontier, per-dataset and pooled,
                        with multi-seed mean +/- std and bootstrap CIs

The through-line is protocol §6's central control: **the same generations feed every
method and every N**. Stage A draws them once; nothing downstream re-generates. That both
isolates generation latency out of the method comparison and makes the N-sweep a cache
truncation rather than N separate runs.
"""

# Lightweight, dependency-free surfaces load eagerly. The Stage-A generators pull in the
# full TruthTorchLM/ML stack, so they load lazily via __getattr__ below -- that keeps
# `import hc_benchmark` (and the readiness report, config, generator registry) usable
# without torch installed, e.g. to check what needs preparing before a cluster run.
from .config import BenchmarkConfig, load_config
from .cache import GenerationCache
from .generators import GENERATORS, GeneratorSpec, get_generator, generators_for

__all__ = [
    "BenchmarkConfig",
    "load_config",
    "GenerationCache",
    "generate_stage_a",
    "generate_stage_a_local",
    "GENERATORS",
    "GeneratorSpec",
    "get_generator",
    "generators_for",
]

_LAZY = {
    "generate_stage_a": ".stage_a_generate",
    "generate_stage_a_local": ".stage_a_generate",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
