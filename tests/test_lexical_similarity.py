"""Lexical Similarity — the pure pairwise-averaging arithmetic (ROUGE-L port).

The ROUGE-L scorer needs `evaluate`; what is porting-sensitive and tested here without it
is the pairwise averaging: identical answers -> 1.0, a lone answer -> 1.0, and the mean is
over all pairs. A fake scorer with a known similarity matrix makes the arithmetic exact.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load():
    dotted = "TruthTorchLM.truth_methods.lexical_similarity"
    if dotted in sys.modules:
        return sys.modules[dotted]
    for name in ("torch", "transformers"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda n: object
            sys.modules[name] = stub
    for pkg in ("TruthTorchLM", "TruthTorchLM.truth_methods"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    sys.modules["TruthTorchLM.truth_methods.truth_method"] = types.ModuleType(
        "TruthTorchLM.truth_methods.truth_method"
    )
    sys.modules["TruthTorchLM.truth_methods.truth_method"].TruthMethod = object
    gen = types.ModuleType("TruthTorchLM.generation")
    gen.sample_generations_hf_local = lambda *a, **k: None
    gen.sample_generations_api = lambda *a, **k: None
    sys.modules["TruthTorchLM.generation"] = gen

    spec = importlib.util.spec_from_file_location(dotted, SRC / "TruthTorchLM/truth_methods/lexical_similarity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


les = _load()
_mean_pairwise_rougeL = les._mean_pairwise_rougeL


class _FakeRouge:
    """Returns a preset ROUGE-L for a given unordered pair, keyed by frozenset of texts."""

    def __init__(self, table):
        self.table = table

    def compute(self, predictions, references, rouge_types):
        key = frozenset((predictions[0], references[0]))
        return {"rougeL": self.table.get(key, 1.0 if predictions[0] == references[0] else 0.0)}


def test_single_generation_is_maximally_self_consistent():
    assert _mean_pairwise_rougeL(["only one"], _FakeRouge({})) == 1.0


def test_empty_is_one_by_convention():
    assert _mean_pairwise_rougeL([], _FakeRouge({})) == 1.0


def test_identical_generations_average_to_one():
    """The confident case: every sample rewords the same answer."""
    texts = ["Paris", "Paris", "Paris"]
    assert _mean_pairwise_rougeL(texts, _FakeRouge({})) == pytest.approx(1.0)


def test_all_distinct_generations_average_to_zero():
    """The uncertain case: no overlap between any pair."""
    texts = ["Paris", "London", "Berlin"]
    rouge = _FakeRouge({})  # unknown pairs of distinct strings -> 0.0
    assert _mean_pairwise_rougeL(texts, rouge) == pytest.approx(0.0)


def test_mean_is_over_all_pairs():
    """3 samples -> 3 unordered pairs; the mean is their average ROUGE-L."""
    texts = ["a", "b", "c"]
    table = {
        frozenset(("a", "b")): 0.9,
        frozenset(("a", "c")): 0.3,
        frozenset(("b", "c")): 0.6,
    }
    expected = (0.9 + 0.3 + 0.6) / 3
    assert _mean_pairwise_rougeL(texts, _FakeRouge(table)) == pytest.approx(expected)


def test_higher_similarity_is_higher_truth_value():
    """Orientation: more overlap must score higher (the truth-value sign)."""
    high = _mean_pairwise_rougeL(["cats are nice", "cats are nice"], _FakeRouge({}))
    low = _mean_pairwise_rougeL(["cats are nice", "dogs bark loudly"], _FakeRouge({}))
    assert high > low
