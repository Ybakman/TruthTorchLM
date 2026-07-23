"""DisAAD — the evidential scoring math (Stage 2) and the teacher->student wiring (Stage 1).

The proxy forward pass needs a trained checkpoint + GPUs (cluster only), so what is tested
here is what can be verified anywhere:
  * the evidential uncertainty measures, pinned against the source's `metrics.get_eu`
    formulas recomputed by hand, and
  * the training command construction, which encodes the teacher/student mapping onto the
    official scripts -- verifiable dry, without any GPU.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.special import digamma, softmax

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _load_scoring():
    dotted = "TruthTorchLM.truth_methods.disaad"
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
    tm = types.ModuleType("TruthTorchLM.truth_methods.truth_method")
    tm.TruthMethod = type("TruthMethod", (), {"__init__": lambda self: None})
    sys.modules["TruthTorchLM.truth_methods.truth_method"] = tm
    spec = importlib.util.spec_from_file_location(dotted, SRC / "TruthTorchLM/truth_methods/disaad.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_train():
    dotted = "hc_benchmark.disaad_train"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/disaad_train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


dis = _load_scoring()
train = _load_train()


class TestEvidentialMeasures:
    def test_epistemic_matches_source_formula(self):
        """metrics.get_eu('eu'): k / (sum(max(0, top-k)) + k)."""
        logits = np.array([3.0, 1.0, -2.0, 0.5, 4.0])
        k = 2
        top = np.sort(logits)[-k:]  # [3, 4]
        expected = k / (np.sum(np.maximum(0, top)) + k)
        assert dis.evidential_epistemic(logits, k) == pytest.approx(expected)

    def test_epistemic_is_larger_when_evidence_is_scarce(self):
        """Small logits (little evidence) -> high epistemic uncertainty."""
        scarce = dis.evidential_epistemic(np.array([0.1, 0.05, 0.2]), 2)
        abundant = dis.evidential_epistemic(np.array([50.0, 40.0, 30.0]), 2)
        assert scarce > abundant

    def test_aleatoric_matches_source_edl_formula(self):
        logits = np.array([5.0, 3.0, 2.0, 1.0])
        k = 3
        alpha = np.sort(logits)[-k:].reshape(1, -1)  # top-3 as evidence
        alpha_0 = alpha.sum(axis=1, keepdims=True)
        expected = float((-(alpha / alpha_0) * (digamma(alpha + 1) - digamma(alpha_0 + 1))).sum())
        assert dis.evidential_aleatoric(logits, k) == pytest.approx(expected)

    def test_msp_and_entropy(self):
        logits = np.array([2.0, 1.0, 0.0])
        p = softmax(logits)
        assert dis.max_softmax_probability(logits) == pytest.approx(p.max())
        assert dis.softmax_entropy(logits) == pytest.approx(-np.sum(p * np.log(p + 1e-10)))

    def test_k_larger_than_vocab_raises(self):
        with pytest.raises(ValueError, match="< k"):
            dis.evidential_epistemic(np.array([1.0, 2.0]), 5)


class TestDisAADScorerConfig:
    def test_mode_validation(self):
        with pytest.raises(ValueError, match="au/eu/msp/entropy"):
            dis.DisAAD(mode="bogus")

    def test_single_instance_no_extra_target_calls(self):
        """DisAAD consumes the already-produced answer + one proxy pass -- no target calls."""
        assert dis.DisAAD.NUM_TARGET_CALLS == 1
        assert dis.DisAAD.REQUIRES_SAMPLED_TEXT is False

    def test_scoring_without_a_trained_proxy_raises_clearly(self):
        method = dis.DisAAD(mode="au")  # no proxy loaded
        with pytest.raises(RuntimeError, match="trained proxy"):
            method._score_with_proxy("prompt", "response")

    def test_requires_training_flag_and_is_ready(self):
        """DisAAD advertises that it needs a prep step, and is_ready reflects it."""
        assert dis.DisAAD.REQUIRES_TRAINING is True
        method = dis.DisAAD(mode="au")           # nothing loaded, no path
        assert method.is_ready() is False

    def test_is_ready_true_when_proxy_path_has_a_manifest(self, tmp_path):
        from hc_benchmark.disaad_train import DisAADTrainingConfig, write_training_manifest

        write_training_manifest(
            DisAADTrainingConfig(teacher_model="gpt-4o-mini", proxy_output_path=str(tmp_path)),
            str(tmp_path),
        )
        method = dis.DisAAD(mode="au", proxy_path=str(tmp_path))
        assert method.is_ready() is True
        assert dis.DisAAD.is_trained(str(tmp_path)) is True


class TestTeacherStudentWiring:
    def test_api_teacher_maps_to_the_api_data_path(self):
        cfg = train.DisAADTrainingConfig(teacher_model="gpt-4o-mini", teacher_is_api=True,
                                         student_model="llama-3.2-3b")
        cmd = train.build_data_builder_command(cfg)
        assert "--what_to_do" in cmd and cmd[cmd.index("--what_to_do") + 1] == "api"
        assert "--api_model_name" in cmd and cmd[cmd.index("--api_model_name") + 1] == "gpt-4o-mini"
        assert "--base_model_name" not in cmd

    def test_local_teacher_maps_to_the_local_data_path(self):
        cfg = train.DisAADTrainingConfig(teacher_model="llama2-70b", teacher_is_api=False)
        cmd = train.build_data_builder_command(cfg)
        assert cmd[cmd.index("--what_to_do") + 1] == "local"
        assert cmd[cmd.index("--base_model_name") + 1] == "llama2-70b"

    def test_train_command_maps_teacher_to_target_and_student_to_scoring(self):
        """The core wiring: teacher -> --target_model_name, student -> --scoring_model_name."""
        cfg = train.DisAADTrainingConfig(teacher_model="gpt-4o-mini", student_model="llama-3.2-3b")
        cmd = train.build_train_command(cfg)
        assert cmd[cmd.index("--target_model_name") + 1] == "gpt-4o-mini"    # teacher
        assert cmd[cmd.index("--scoring_model_name") + 1] == "llama-3.2-3b"  # student/proxy
        assert cmd[cmd.index("--output_path") + 1] == cfg.proxy_output_path

    def test_train_proxy_dry_run_returns_both_stage_commands_without_running(self):
        cfg = train.DisAADTrainingConfig(teacher_model="gpt-4o-mini")
        # dry_run must not require the submodule to be present? It does check; ensure it's there.
        out = train.train_proxy(cfg, dry_run=True)
        assert set(out) == {"data_builder", "train_disaad"}
        assert out["data_builder"][0] == "python"
        assert "train_disaad.py" in out["train_disaad"][1]
