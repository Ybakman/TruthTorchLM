"""OOD / density gate — the E-cube classifier and the KB-aligned PCA gate.

sklearn is available, so the gate is tested end-to-end with an injected embedder (a fake
that maps texts to 2-D points by a lookup) -- no sentence-transformers, no network, but the
real PCA + neighbor classifiers. The E-cube classifier is pinned directly against its box
semantics.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    dotted = "hc_benchmark.ood_gate"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/ood_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


ood = _load()
EpsilonCubeNeighborsClassifier = ood.EpsilonCubeNeighborsClassifier
PCAGate = ood.PCAGate


class TestEpsilonCube:
    def test_point_inside_the_box_takes_the_neighbor_label(self):
        clf = EpsilonCubeNeighborsClassifier(sides=[1.0, 1.0], outlier_label=0)
        clf.fit(np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([1, 1]))
        # (0.5, 0.5) is within +/-1 of the first training point -> in-domain (1)
        assert clf.predict(np.array([[0.5, 0.5]])) == [1]

    def test_point_outside_every_box_is_the_outlier_label(self):
        clf = EpsilonCubeNeighborsClassifier(sides=[1.0, 1.0], outlier_label=0)
        clf.fit(np.array([[0.0, 0.0]]), np.array([1]))
        assert clf.predict(np.array([[5.0, 5.0]])) == [0]

    def test_box_is_axis_aligned_L_infinity_not_euclidean(self):
        """Both coordinates must be within sides -- a point close in one axis but far in the
        other is still outside (this is the cube, not the ball)."""
        clf = EpsilonCubeNeighborsClassifier(sides=[1.0, 1.0], outlier_label=0)
        clf.fit(np.array([[0.0, 0.0]]), np.array([1]))
        assert clf.predict(np.array([[0.5, 5.0]])) == [0]  # near in x, far in y

    def test_majority_vote_among_in_box_neighbors(self):
        clf = EpsilonCubeNeighborsClassifier(sides=[2.0], outlier_label=0)
        clf.fit(np.array([[0.0], [0.1], [0.2]]), np.array([1, 1, 2]))
        assert clf.predict(np.array([[0.0]])) == [1]  # two 1s vs one 2


class _FakeEmbedder:
    """Maps known texts to preset 2-D points; unknown texts to a far-away point."""

    def __init__(self, table, default=(100.0, 100.0)):
        self.table = table
        self.default = default

    def __call__(self, texts):
        return np.array([self.table.get(t, self.default) for t in texts], dtype=float)


class TestPCAGate:
    def _kb(self):
        # A tight in-domain cluster near the origin.
        return {
            f"kb{i}": (x, y)
            for i, (x, y) in enumerate([(0.0, 0.0), (0.1, 0.1), (-0.1, 0.05),
                                        (0.05, -0.1), (0.0, 0.1)])
        }

    def test_in_domain_query_passes_out_of_domain_query_is_gated(self):
        kb = self._kb()
        table = dict(kb)
        table["near"] = (0.02, 0.0)      # close to the KB cluster
        table["far"] = (50.0, -40.0)     # nowhere near it
        gate = PCAGate(embed_fn=_FakeEmbedder(table), method="eball", radius=2.0,
                       scale=False)       # scaling a tiny synthetic KB is unstable; off here
        gate.fit(list(kb.keys()))
        assert gate.is_in_domain("near") is True
        assert gate.is_in_domain("far") is False

    def test_ecube_method_also_gates(self):
        kb = self._kb()
        table = dict(kb)
        table["near"] = (0.02, 0.0)
        table["far"] = (80.0, 80.0)
        gate = PCAGate(embed_fn=_FakeEmbedder(table), method="ecube", radius=2.0, scale=False)
        gate.fit(list(kb.keys()))
        assert gate.predict(["near", "far"]) == [gate.IN_DOMAIN, gate.OUT_OF_DOMAIN]

    def test_predict_before_fit_raises(self):
        gate = PCAGate(embed_fn=_FakeEmbedder({}))
        with pytest.raises(RuntimeError, match="not fitted"):
            gate.predict(["q"])

    def test_bad_method_raises(self):
        with pytest.raises(ValueError, match="eball.*ecube"):
            PCAGate(embed_fn=_FakeEmbedder({}), method="knn")

    def test_it_is_an_input_gate_not_a_truth_method(self):
        """Sanity on the architectural role: the gate scores the *query*, so it exposes
        is_in_domain, not a forward()/truth_value over a generation."""
        gate = PCAGate(embed_fn=_FakeEmbedder({}))
        assert hasattr(gate, "is_in_domain")
        assert not hasattr(gate, "forward_api")
