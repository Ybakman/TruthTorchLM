"""Benchmark configuration and its content-hash identity (protocol §5 measurement hygiene).

Protocol §5 requires that every latency number be reported alongside the exact conditions
that produced it -- target model, hardware/endpoint, output token count, batch size,
concurrency level, sampling params. The reliable way to guarantee that is to make the
config *part of the artifact's identity*: every cache file and result file is keyed by a
hash of the config that produced it, so a number can never be silently compared across
different decoding settings. This records the conditions by construction rather than by
the operator remembering to.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import yaml

__all__ = ["DecodingConfig", "BenchmarkConfig", "load_config"]


@dataclass(frozen=True)
class DecodingConfig:
    """Decoding params. Held constant across methods (protocol §6), so part of the key.

    ``temperature``/``top_p`` here govern the *primary* answer. ``sample_temperature`` /
    ``sample_top_p`` govern the multi-sample draws, which is the knob the consistency
    family actually depends on and which must be identical across those methods.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9


@dataclass(frozen=True)
class BenchmarkConfig:
    """A full benchmark run's parameters.

    Note the two distinct notions of "how many samples". ``n_max`` is what Stage A
    actually draws and caches (once). ``n_sweep`` is the set of budgets Stage C evaluates
    by *truncating* that cache -- so every N in the sweep reuses the identical draws and
    the frontier is not confounded by re-sampling. ``n_sweep`` values above ``n_max`` are
    therefore meaningless and rejected.
    """

    dataset: str
    generator: str
    generator_backend: str = "litellm"  # "litellm" (API) or "huggingface" (local)
    n_max: int = 20
    n_sweep: tuple = (1, 3, 5, 10, 20)
    seeds: tuple = (0, 1, 2)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    size_of_data: float = 1.0

    # Latency (protocol §5).
    execution_modes: tuple = ("serial", "concurrent")
    max_workers: int = 8
    sla_budgets_ms: tuple = (500.0, 1000.0, 2000.0)
    latency_trials: int = 100
    latency_warmup: int = 5

    # Correctness labels (protocol §3 -- run both, report sensitivity).
    correctness_criteria: tuple = ("llm_judge", "bleurt")
    bleurt_threshold: float = 0.5  # matches DisAAD's --thres_gt 0.5

    # Measurement-hygiene metadata: recorded verbatim into every artifact.
    hardware: str = "unspecified"
    api_endpoint: str = "unspecified"
    notes: str = ""

    def model_string(self) -> str:
        """The actual model string to pass to the backend.

        If ``generator`` is a registry key (e.g. ``claude-opus-4-8``), returns the
        provider-qualified id LiteLLM/HF expects (``anthropic/claude-opus-4-8``);
        otherwise returns ``generator`` verbatim, so an ad-hoc id still works.
        """
        spec = self.resolve_generator()
        return spec.model_id if spec is not None else self.generator

    def resolved_backend(self) -> str:
        """Backend for this generator, taken from the registry when available."""
        spec = self.resolve_generator()
        return spec.backend if spec is not None else self.generator_backend

    def resolve_generator(self):
        """The :class:`GeneratorSpec` for this config's generator, if it's in the registry.

        Lets a config name a target by friendly key (``claude-opus-4-8``) and have the
        backend, LiteLLM/HF model id, access level, and reasoning-trace policy filled in
        from one place. Returns None for an ad-hoc model id not in the registry, in which
        case ``generator``/``generator_backend`` are used verbatim.
        """
        from .generators import GENERATORS

        return GENERATORS.get(self.generator)

    def __post_init__(self):
        bad = [n for n in self.n_sweep if n > self.n_max]
        if bad:
            raise ValueError(
                f"n_sweep values {bad} exceed n_max={self.n_max}. Stage A only caches "
                f"n_max samples; higher budgets can't be served by truncation. Raise "
                f"n_max or drop those values."
            )
        if not self.n_sweep:
            raise ValueError("n_sweep is empty -- nothing to evaluate.")

    def content_hash(self) -> str:
        """Stable short hash over the fields that change the *data* (not just labels).

        Excludes the hygiene metadata and SLA budgets -- those annotate results but don't
        change generations or scores, so a note edit shouldn't invalidate a cache.
        """
        material = {
            "dataset": self.dataset,
            "generator": self.generator,
            "generator_backend": self.generator_backend,
            "n_max": self.n_max,
            "decoding": asdict(self.decoding),
            "size_of_data": self.size_of_data,
        }
        blob = json.dumps(material, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_hash"] = self.content_hash()
        return d


def load_config(path: str) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a YAML file (see hc_benchmark/configs/)."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    decoding = DecodingConfig(**(raw.pop("decoding", {}) or {}))
    for key in ("n_sweep", "seeds", "execution_modes", "sla_budgets_ms", "correctness_criteria"):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])
    return BenchmarkConfig(decoding=decoding, **raw)
