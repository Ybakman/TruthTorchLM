"""Method readiness — the "do I need to prepare this first, and is it done?" signal.

Answers the two usability questions directly: which methods need a prep step (DisAAD needs
a trained proxy; the OOD gate needs a KB fit), and how a user knows the prep is finished (a
training manifest / a completed fit). Loaded standalone -- readiness is dependency-light.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(dotted, rel):
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


train = _load("hc_benchmark.disaad_train", "hc_benchmark/disaad_train.py")
readiness = _load("hc_benchmark.readiness", "hc_benchmark/readiness.py")
ood = _load("hc_benchmark.ood_gate", "hc_benchmark/ood_gate.py")


class TestTrainingManifest:
    def test_untrained_dir_is_not_ready(self, tmp_path):
        assert train.is_proxy_trained(str(tmp_path)) is False
        assert train.read_training_manifest(str(tmp_path)) is None

    def test_manifest_written_on_completion_marks_ready(self, tmp_path):
        cfg = train.DisAADTrainingConfig(teacher_model="gpt-4o-mini",
                                         student_model="llama-3.2-3b",
                                         proxy_output_path=str(tmp_path))
        train.write_training_manifest(cfg, str(tmp_path))
        assert train.is_proxy_trained(str(tmp_path)) is True
        m = train.read_training_manifest(str(tmp_path))
        assert m["teacher_model"] == "gpt-4o-mini"
        assert m["student_model"] == "llama-3.2-3b"
        assert "trained_at" in m

    def test_a_dir_with_weights_but_no_manifest_is_still_not_ready(self, tmp_path):
        """A half-written / externally-trained proxy dir must not read as ready."""
        (tmp_path / "model.safetensors").write_bytes(b"not a real checkpoint")
        assert train.is_proxy_trained(str(tmp_path)) is False


class TestReadinessReport:
    def test_inference_methods_are_always_ready(self):
        rows = readiness.readiness_report()
        inf = [r for r in rows if r.kind == "inference"]
        assert inf and all(r.ready for r in inf)
        names = {r.name for r in inf}
        assert {"DiscreteSemanticEntropy", "SPUQ", "IUQ",
                "NeighborhoodConsistencyBelief"} <= names

    def test_disaad_not_ready_without_a_proxy(self):
        rows = {r.name: r for r in readiness.readiness_report()}
        assert rows["DisAAD"].ready is False
        assert "Train a proxy" in rows["DisAAD"].how_to_prepare

    def test_disaad_flips_to_ready_once_the_manifest_exists(self, tmp_path):
        cfg = train.DisAADTrainingConfig(teacher_model="claude-opus-4-8",
                                         proxy_output_path=str(tmp_path))
        train.write_training_manifest(cfg, str(tmp_path))
        rows = {r.name: r for r in readiness.readiness_report(proxy_path=str(tmp_path))}
        assert rows["DisAAD"].ready is True
        assert "claude-opus-4-8" in rows["DisAAD"].detail   # teacher shown so it's traceable

    def test_ood_gate_ready_only_after_fit(self):
        rows_unfitted = {r.name: r for r in readiness.readiness_report()}
        assert rows_unfitted["OOD-PCA gate"].ready is False

        gate = ood.PCAGate(embed_fn=lambda xs: [[0.0, 0.0] for _ in xs], scale=False)
        assert gate.is_ready() is False
        gate.fit(["a", "b", "c"])
        assert gate.is_ready() is True
        rows_fitted = {r.name: r for r in readiness.readiness_report(ood_gate=gate)}
        assert rows_fitted["OOD-PCA gate"].ready is True
