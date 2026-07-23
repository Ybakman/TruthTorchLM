"""Discrete Semantic Entropy — the ported cluster-assignment entropy.

The clustering (NLI entailment) needs the ML stack and a model, so it's exercised by the
smoke run. What is tested here without any model is the part that had to be *ported
correctly*: the entropy aggregation and the cluster->semantic-id mapping. These are pinned
against the worked examples in the source's own docstring, so a porting error (wrong base,
dropped normalization, off-by-one on ids) fails loudly.

The two pure functions are loaded standalone — importing the full method pulls in
transformers, which isn't needed to check arithmetic.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load_pure_functions():
    """Load cluster_assignment_entropy + _semantic_ids_from_clusters without transformers.

    The module top-imports torch/transformers, so exec it against a stubbed set of those
    names — the two functions we test touch only numpy.
    """
    import types

    for name in ("torch", "transformers"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda n: object  # any attribute access returns a dummy
            sys.modules[name] = stub
    # TruthTorchLM.utils / generation are imported by the module; stub just enough.
    for dotted in ("TruthTorchLM", "TruthTorchLM.utils"):
        if dotted not in sys.modules:
            m = types.ModuleType(dotted)
            m.__path__ = []
            sys.modules[dotted] = m
    sys.modules["TruthTorchLM.utils"].bidirectional_entailment_clustering = lambda *a, **k: []
    tm = types.ModuleType("TruthTorchLM.truth_methods")
    tm.__path__ = []
    sys.modules["TruthTorchLM.truth_methods"] = tm
    sys.modules["TruthTorchLM.truth_methods.truth_method"] = types.ModuleType(
        "TruthTorchLM.truth_methods.truth_method"
    )
    sys.modules["TruthTorchLM.truth_methods.truth_method"].TruthMethod = object
    gen = types.ModuleType("TruthTorchLM.generation")
    gen.sample_generations_hf_local = lambda *a, **k: None
    gen.sample_generations_api = lambda *a, **k: None
    sys.modules["TruthTorchLM.generation"] = gen

    # Load under the real dotted name so the module's `.truth_method` / `..generation`
    # relative imports resolve against the stubbed parent packages above.
    dotted = "TruthTorchLM.truth_methods.discrete_semantic_entropy"
    path = SRC / "TruthTorchLM/truth_methods/discrete_semantic_entropy.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


dse = _load_pure_functions()
cluster_assignment_entropy = dse.cluster_assignment_entropy
_semantic_ids_from_clusters = dse._semantic_ids_from_clusters


class TestClusterAssignmentEntropy:
    def test_docstring_worked_example(self):
        """The source docstring's own example: ids [0,1,2,1] -> p=[1/4,2/4,1/4]."""
        p = np.array([1 / 4, 2 / 4, 1 / 4])
        expected = -(p * np.log(p)).sum()
        assert cluster_assignment_entropy([0, 1, 2, 1]) == pytest.approx(expected)

    def test_all_samples_agree_is_zero_entropy(self):
        """One cluster -> perfectly certain -> entropy 0 -> truth value is its max."""
        assert cluster_assignment_entropy([0, 0, 0, 0, 0]) == pytest.approx(0.0)

    def test_all_samples_differ_is_maximum_entropy(self):
        """N singleton clusters -> uniform over N -> entropy log N."""
        assert cluster_assignment_entropy([0, 1, 2, 3, 4]) == pytest.approx(math.log(5))

    def test_entropy_distinguishes_splits_that_the_count_collapses(self):
        """The reason DSE exists rather than NumSemanticSet: same cluster *count* (2),
        different assignment distribution, different entropy."""
        four_one = cluster_assignment_entropy([0, 0, 0, 0, 1])   # 4/1 split
        three_two = cluster_assignment_entropy([0, 0, 0, 1, 1])  # 3/2 split
        assert three_two > four_one  # more balanced = more uncertain
        # NumSemanticSet would report |C| = 2 for both, hiding this.

    def test_probabilities_normalize(self):
        # A non-contiguous id set still normalizes (bincount fills the gap with a 0 count,
        # which the nonzero guard drops).
        assert cluster_assignment_entropy([0, 2, 2]) == pytest.approx(
            -(np.array([1 / 3, 2 / 3]) * np.log([1 / 3, 2 / 3])).sum()
        )


class TestSemanticIdMapping:
    def test_maps_clusters_to_per_sample_ids_in_generation_order(self):
        """TTLM returns groups of texts; the ported entropy needs one id per sample in the
        original order. This is where an ordering bug would silently corrupt the entropy."""
        texts = ["Paris", "It's Paris", "London", "Paris."]
        clusters = [["Paris", "It's Paris", "Paris."], ["London"]]
        ids = _semantic_ids_from_clusters(clusters, texts)
        assert ids == [0, 0, 1, 0]

    def test_singleton_clusters(self):
        texts = ["a", "b", "c"]
        clusters = [["a"], ["b"], ["c"]]
        assert _semantic_ids_from_clusters(clusters, texts) == [0, 1, 2]

    def test_round_trip_into_entropy(self):
        """End to end on the pure path: clusters -> ids -> entropy, agreeing/​disagreeing."""
        agree = _semantic_ids_from_clusters([["a", "b", "c"]], ["a", "b", "c"])
        assert cluster_assignment_entropy(agree) == pytest.approx(0.0)
        disagree = _semantic_ids_from_clusters([["a"], ["b"], ["c"]], ["a", "b", "c"])
        assert cluster_assignment_entropy(disagree) == pytest.approx(math.log(3))
