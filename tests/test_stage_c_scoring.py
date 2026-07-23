"""Stage C scoring against a fake TruthMethod (no live model).

Proves the cache -> score path end to end, and the property that makes the whole harness
trustworthy: at score time a method receives its samples from the cache and does **not**
generate. A method that tried to call a target here would fail, because the "model" passed
in is an inert sentinel -- which is exactly the guarantee we want to hold.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _load(dotted, relpath):
    if dotted in sys.modules:
        return sys.modules[dotted]
    for pkg_name, pkg_path in (("hc_benchmark", REPO_ROOT / "hc_benchmark"),):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


config_mod = _load("hc_benchmark.config", "hc_benchmark/config.py")
cache_mod = _load("hc_benchmark.cache", "hc_benchmark/cache.py")
stage_c = _load("hc_benchmark.stage_c_score", "hc_benchmark/stage_c_score.py")

BenchmarkConfig = config_mod.BenchmarkConfig
GenerationCache = cache_mod.GenerationCache


class FakeConsistencyMethod:
    """A stand-in TruthMethod: uncertainty = number of distinct samples.

    Requires no model and no logprobs. It reads only the cached text, which is the whole
    point -- if the harness tried to make it generate, there is nothing here to generate
    with. ``number_of_generations`` mirrors the real interface.
    """

    def __init__(self, number_of_generations=5):
        self.number_of_generations = number_of_generations
        self.saw_model = None

    def __call__(self, model=None, sampled_generations_dict=None, **kwargs):
        self.saw_model = model  # recorded so the test can assert it was never used to call out
        samples = sampled_generations_dict["generated_texts"]
        distinct = len(set(samples))
        # More agreement (fewer distinct) -> higher truth value.
        truth = 1.0 / distinct if distinct else 0.0
        return {"truth_value": truth, "normalized_truth_value": truth}


INERT_MODEL = "sentinel::no-network"


@pytest.fixture
def populated_cache(tmp_path):
    cfg = BenchmarkConfig(dataset="d", generator="g", n_max=5, n_sweep=(1, 3, 5))
    cache = GenerationCache(str(tmp_path), cfg, seed=0)
    cache.write(
        [
            # item 0: samples all agree -> low uncertainty
            {"item_id": 0, "question": "q0", "context": "", "ground_truths": ["a"],
             "primary_answer": "a", "samples": ["a", "a", "a", "a", "a"],
             "stratum": None, "outcome_type": "factual_error"},
            # item 1: samples all differ -> high uncertainty
            {"item_id": 1, "question": "q1", "context": "", "ground_truths": ["b"],
             "primary_answer": "b", "samples": ["b", "c", "d", "e", "f"],
             "stratum": None, "outcome_type": "factual_error"},
        ]
    )
    return cache


def test_scoring_reads_cache_and_never_calls_the_model(populated_cache):
    method = FakeConsistencyMethod()
    results = stage_c.score_stage_c(
        populated_cache, [method], model=INERT_MODEL, n_sweep=(1, 5), collect_timing=False
    )
    # The sentinel reached the method but was only ever passed through, never used to
    # generate -- proof the cache, not the target, supplied the samples.
    assert method.saw_model == INERT_MODEL

    scores = results["FakeConsistencyMethod"]
    assert set(scores) == {1, 5}
    # At N=5 the agreeing item scores higher (more certain) than the disagreeing one.
    tv = scores[5]["truth_values"]
    assert tv[0] > tv[1]


def test_n_sweep_truncates_the_same_draw(populated_cache):
    method = FakeConsistencyMethod()
    results = stage_c.score_stage_c(
        populated_cache, [method], model=INERT_MODEL, n_sweep=(1, 3, 5), collect_timing=False
    )
    scores = results["FakeConsistencyMethod"]
    # Item 1's samples are all distinct, so distinct-count == N: uncertainty rises with N,
    # and it does so over a *prefix* of one cached draw, not fresh samples.
    assert scores[1]["truth_values"][1] == pytest.approx(1.0)      # 1 distinct
    assert scores[3]["truth_values"][1] == pytest.approx(1 / 3)    # 3 distinct
    assert scores[5]["truth_values"][1] == pytest.approx(1 / 5)    # 5 distinct


def test_timing_is_collected_per_item_when_requested(populated_cache):
    method = FakeConsistencyMethod()
    results = stage_c.score_stage_c(
        populated_cache, [method], model=INERT_MODEL, n_sweep=(5,), collect_timing=True
    )
    timing = results["FakeConsistencyMethod"][5]["timing_ms"]
    assert len(timing) == 2  # one auxiliary-compute measurement per item
    assert all(t >= 0 for t in timing)
