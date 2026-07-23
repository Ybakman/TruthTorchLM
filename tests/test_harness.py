"""The Stage A-D harness plumbing: config identity, the shared cache, seed aggregation.

These exercise the parts that must be correct for the *comparison* to be valid, not the
model-facing parts (which need a live target and are covered by the smoke run):

* the config content-hash keys artifacts, so a decoding change can't silently reuse an
  old cache (protocol §5 hygiene);
* the cache round-trips nested samples and, crucially, truncates to serve the N-sweep
  from one draw (protocol §6);
* seed aggregation produces the mean +/- std the protocol requires over single runs (§6).

The stage modules that import TruthTorchLM are loaded standalone (config/cache/stage_d
depend only on pandas/numpy/yaml), mirroring conftest's approach.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(dotted, relpath):
    if dotted in sys.modules:
        return sys.modules[dotted]
    # Register the package shell so intra-package relative imports resolve.
    if "hc_benchmark" not in sys.modules:
        import types

        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


config_mod = _load("hc_benchmark.config", "hc_benchmark/config.py")
cache_mod = _load("hc_benchmark.cache", "hc_benchmark/cache.py")
stage_d = _load("hc_benchmark.stage_d_evaluate", "hc_benchmark/stage_d_evaluate.py")

BenchmarkConfig = config_mod.BenchmarkConfig
DecodingConfig = config_mod.DecodingConfig
load_config = config_mod.load_config
GenerationCache = cache_mod.GenerationCache


class TestConfigIdentity:
    def test_same_content_same_hash(self):
        a = BenchmarkConfig(dataset="bioasq", generator="gpt-4o-mini")
        b = BenchmarkConfig(dataset="bioasq", generator="gpt-4o-mini")
        assert a.content_hash() == b.content_hash()

    def test_decoding_change_changes_the_hash(self):
        """A different sampling temperature is a different experiment -- it must not reuse
        another run's cache."""
        a = BenchmarkConfig(dataset="bioasq", generator="g")
        b = BenchmarkConfig(dataset="bioasq", generator="g",
                            decoding=DecodingConfig(sample_temperature=0.9))
        assert a.content_hash() != b.content_hash()

    def test_hygiene_notes_do_not_change_the_hash(self):
        """A hardware note or SLA budget annotates results but doesn't change the data,
        so editing it must not invalidate an expensive generation cache."""
        a = BenchmarkConfig(dataset="bioasq", generator="g", notes="run 1")
        b = BenchmarkConfig(dataset="bioasq", generator="g", notes="run 2",
                            hardware="A100", sla_budgets_ms=(250.0,))
        assert a.content_hash() == b.content_hash()

    def test_n_sweep_above_n_max_is_rejected(self):
        """You cannot serve N=20 by truncating a cache that only holds 10 draws."""
        with pytest.raises(ValueError, match="exceed n_max"):
            BenchmarkConfig(dataset="d", generator="g", n_max=10, n_sweep=(1, 5, 20))

    def test_yaml_round_trip(self, tmp_path):
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(
            "dataset: medqa\ngenerator: gpt-4o-mini\nn_max: 5\nn_sweep: [1, 5]\n"
            "seeds: [0, 1]\ndecoding:\n  sample_temperature: 0.5\n"
        )
        cfg = load_config(str(cfg_path))
        assert cfg.dataset == "medqa"
        assert cfg.n_sweep == (1, 5)
        assert cfg.decoding.sample_temperature == 0.5


class TestGenerationCache:
    def _items(self):
        return [
            {"item_id": 0, "question": "q0", "context": "", "ground_truths": ["a0"],
             "primary_answer": "p0", "samples": ["s0", "s1", "s2", "s3", "s4"],
             "stratum": None, "outcome_type": "factual_error"},
            {"item_id": 1, "question": "q1", "context": "ctx", "ground_truths": ["a1", "a1b"],
             "primary_answer": "p1", "samples": ["t0", "t1", "t2", "t3", "t4"],
             "stratum": "medical_mcq", "outcome_type": "factual_error"},
        ]

    def test_write_then_read_round_trips_nested_fields(self, tmp_path):
        cfg = BenchmarkConfig(dataset="d", generator="g", n_max=5, n_sweep=(1, 3, 5))
        cache = GenerationCache(str(tmp_path), cfg, seed=0)
        cache.write(self._items())

        loaded = cache.read()
        assert len(loaded) == 2
        assert loaded[0]["ground_truths"] == ["a0"]
        assert loaded[1]["ground_truths"] == ["a1", "a1b"]
        assert loaded[0]["samples"] == ["s0", "s1", "s2", "s3", "s4"]
        assert loaded[1]["stratum"] == "medical_mcq"

    def test_truncation_serves_the_n_sweep_from_one_draw(self, tmp_path):
        """The heart of the §6 control: read(n=1) and read(n=5) are prefixes of the same
        cached samples, so an N-sweep never re-samples."""
        cfg = BenchmarkConfig(dataset="d", generator="g", n_max=5, n_sweep=(1, 3, 5))
        cache = GenerationCache(str(tmp_path), cfg, seed=0)
        cache.write(self._items())

        assert cache.read(n=1)[0]["samples"] == ["s0"]
        assert cache.read(n=3)[0]["samples"] == ["s0", "s1", "s2"]
        # And the truncations are genuine prefixes of one another.
        assert cache.read(n=5)[0]["samples"][:3] == cache.read(n=3)[0]["samples"]

    def test_requesting_more_samples_than_cached_raises(self, tmp_path):
        cfg = BenchmarkConfig(dataset="d", generator="g", n_max=5, n_sweep=(1, 3, 5))
        cache = GenerationCache(str(tmp_path), cfg, seed=0)
        cache.write(self._items())
        with pytest.raises(ValueError, match="only 5 samples"):
            cache.read(n=10)

    def test_cache_path_encodes_the_config_hash_and_seed(self, tmp_path):
        cfg = BenchmarkConfig(dataset="bioasq", generator="gpt-4o-mini")
        c0 = GenerationCache(str(tmp_path), cfg, seed=0)
        c1 = GenerationCache(str(tmp_path), cfg, seed=1)
        assert c0.path != c1.path
        assert cfg.content_hash() in c0.path.name
        assert "seed0" in c0.path.name

    def test_a_writes_a_self_describing_config_sidecar(self, tmp_path):
        cfg = BenchmarkConfig(dataset="d", generator="g", n_max=5, n_sweep=(1, 3, 5))
        cache = GenerationCache(str(tmp_path), cfg, seed=0)
        cache.write(self._items())
        sidecar = cache.path.with_suffix(".config.json")
        assert sidecar.exists()


class TestStageDStatistics:
    def _auroc(self, correctness, truth_values):
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(correctness, truth_values)

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        # A score correlated with correctness -> AUROC clearly above 0.5.
        correctness = rng.integers(0, 2, 200)
        truth_values = correctness + rng.normal(0, 0.5, 200)
        point, lo, hi = stage_d.bootstrap_ci(self._auroc, correctness, truth_values, n_boot=300)
        assert lo <= point <= hi
        assert 0.5 < point <= 1.0
        assert hi - lo > 0  # a real interval, not a degenerate point

    def test_aggregate_seeds_reports_mean_and_std(self):
        per_seed = [
            {1: {"auroc": 0.70}, 5: {"auroc": 0.80}},
            {1: {"auroc": 0.72}, 5: {"auroc": 0.82}},
            {1: {"auroc": 0.74}, 5: {"auroc": 0.84}},
        ]
        agg = stage_d.aggregate_seeds(per_seed)
        assert agg[1]["auroc"]["mean"] == pytest.approx(0.72)
        assert agg[1]["auroc"]["std"] > 0
        assert agg[1]["auroc"]["n_seeds"] == 3
        assert agg[5]["auroc"]["mean"] == pytest.approx(0.82)

    def test_aggregate_passes_non_numeric_fields_through(self):
        per_seed = [
            {1: {"auroc": 0.7, "harm_operating_point": {"threshold": 0.5}}},
            {1: {"auroc": 0.8, "harm_operating_point": {"threshold": 0.6}}},
        ]
        agg = stage_d.aggregate_seeds(per_seed)
        assert agg[1]["auroc"]["mean"] == pytest.approx(0.75)
        # A dict metric can't be averaged; the first seed's is carried through intact.
        assert agg[1]["harm_operating_point"] == {"threshold": 0.5}


class TestOOD:
    def test_drop_is_oriented_so_positive_always_means_worse(self):
        ood = _load("hc_benchmark.ood", "hc_benchmark/ood.py")
        in_domain = {5: {"auroc": 0.85, "ece": 0.04}}
        out_domain = {5: {"auroc": 0.70, "ece": 0.19}}
        result = ood.ood_degradation(in_domain, out_domain, metrics=("auroc", "ece"))
        # AUROC fell (higher is better) and ECE rose (lower is better); both are worse,
        # so both drops must be positive despite moving in opposite raw directions.
        assert result[5]["auroc"]["drop"] == pytest.approx(0.15)
        assert result[5]["ece"]["drop"] == pytest.approx(0.15)
