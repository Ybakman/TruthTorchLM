"""Capabilities inventory — structure, and a drift guard against the real code.

The inventory is derived from the registry/availability, so most of it can't drift. The
one hand-maintained piece is the method catalog; the torch-gated test at the bottom asserts
it matches the actual truth_methods exports and REQUIRES_TRAINING flags, so it stays honest.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    dotted = "hc_benchmark.capabilities"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/capabilities.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


cap = _load()


class TestStructure:
    def test_all_four_axes_present(self):
        caps = cap.capabilities()
        keys = list(caps)
        assert any(k.startswith("G") for k in keys)
        assert any(k.startswith("D") for k in keys)
        assert any(k.startswith("M") for k in keys)
        assert any(k.startswith("V") for k in keys)

    def test_generators_grouped_by_provider_and_access(self):
        G = cap.capabilities()["G · Generators"]
        assert set(G["By provider"]) <= {"openai", "anthropic", "gemini", "open"}
        assert "anthropic" in G["By provider"]
        # only open models can be proxies
        assert G["By provider"]["open"]["proxies"]
        assert not G["By provider"]["anthropic"]["proxies"]

    def test_every_dataset_is_categorized_in_area_and_format(self):
        """No dataset falls through the area/format buckets (the drift risk for D)."""
        datasets = cap._all_datasets()
        for name in datasets:
            assert cap._dataset_area(name) in {"Health", "General", "Math"}
            assert cap._dataset_format(name)  # non-empty bucket

    def test_health_datasets_are_marked_health(self):
        for name in ["bioasq", "medqa", "mmlu_med", "kqa", "medlfqa"]:
            assert cap._dataset_area(name) == "Health"

    def test_mcq_and_longform_formats(self):
        assert cap._dataset_format("medqa") == "MCQ (constrained output)"
        assert cap._dataset_format("mmlu_med") == "MCQ (constrained output)"
        assert cap._dataset_format("kqa") == "Long-form (claim-level)"
        assert cap._dataset_format("squad_v2") == "Extractive / contextual"

    def test_methods_grouped_by_family_and_preparation(self):
        M = cap.capabilities()["M · UQ methods"]
        fams = M["By family"]
        assert {"VB", "SC", "P", "Input gate"} <= set(fams)
        prep = M["By preparation"]
        # exactly DisAAD needs training; exactly the OOD gate needs a fit
        assert [m[0] for m in prep["needs-training"]] == ["DisAAD"]
        assert [m[0] for m in prep["needs-fit"]] == ["OOD-PCA gate"]

    def test_metrics_axis_lists_the_new_families(self):
        V = cap.capabilities()["V · Metrics"]
        assert "ece" in V["Calibration error"]
        assert "harm_recall" in V["Safety-weighted"]
        assert any("marginal" in m for m in V["Latency (protocol §5)"])


def _require_real_stack():
    """Skip unless the *real* ML stack is importable.

    Sibling test modules stub `torch` into sys.modules to test pure-numpy code, which
    defeats importorskip('torch'). The stub's attributes are all `object`, so a real
    string `__version__` distinguishes it.
    """
    torch = pytest.importorskip("torch")
    if not isinstance(getattr(torch, "__version__", None), str):
        pytest.skip("stubbed torch present (sibling tests) — needs the real ML stack")


class TestDriftGuardAgainstRealCode:
    """Torch-gated: assert the method catalog matches the actual exports + flags."""

    def test_catalog_matches_real_truth_methods(self):
        _require_real_stack()
        import TruthTorchLM.truth_methods as tm

        catalog_truth_methods = {
            name for name, _fam, _src, _prep in cap.METHOD_CATALOG
            if name != "OOD-PCA gate"  # the gate is not a TruthMethod
        }
        for name in catalog_truth_methods:
            assert hasattr(tm, name), f"catalog lists {name} but it is not exported"

    def test_only_disaad_declares_requires_training(self):
        _require_real_stack()
        import TruthTorchLM.truth_methods as tm

        for name, _fam, _src, prep in cap.METHOD_CATALOG:
            if name == "OOD-PCA gate":
                continue
            cls = getattr(tm, name)
            requires = getattr(cls, "REQUIRES_TRAINING", False)
            assert requires == (prep == "needs-training"), \
                f"{name}: REQUIRES_TRAINING={requires} but catalog prep={prep}"
