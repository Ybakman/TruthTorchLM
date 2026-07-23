"""SPUQ — perturbations and the input-weighted output-agreement aggregation.

The target calls and ROUGE scorer need the stack; what is porting-sensitive and tested
here is (1) the perturbation shapes and (2) the aggregation weighting, both of which are
ported from `intuit-ai-research/SPUQ`. A fake generator and injected similarities make the
arithmetic exact and model-free.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load():
    dotted = "TruthTorchLM.truth_methods.spuq"
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

    class _TM:  # minimal TruthMethod stand-in
        REQUIRES_NORMALIZATION = True

        def __init__(self):
            pass

    tm.TruthMethod = _TM
    sys.modules["TruthTorchLM.truth_methods.truth_method"] = tm

    spec = importlib.util.spec_from_file_location(dotted, SRC / "TruthTorchLM/truth_methods/spuq.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


spuq_mod = _load()
SPUQ = spuq_mod.SPUQ
QUESTION = [{"role": "user", "content": "What is the capital of France?"}]


class TestPerturbations:
    def test_system_message_is_prepended_and_count_matches(self):
        out = spuq_mod._perturb_system_message(QUESTION, 1.0, 3, np.random.default_rng(0))
        assert len(out) == 3
        for msgs, temp in out:
            assert msgs[0]["role"] == "system"
            assert msgs[1] == QUESTION[0]  # original question preserved after the system msg
            assert temp == 1.0

    def test_dummy_token_modifies_the_last_message_only(self):
        out = spuq_mod._perturb_dummy_token(QUESTION, 1.0, 4, np.random.default_rng(1))
        assert len(out) == 4
        for msgs, _ in out:
            assert "capital of France" in msgs[-1]["content"]  # question still there
            assert msgs[-1]["content"] != QUESTION[0]["content"]  # but perturbed

    def test_temperature_jitter_stays_in_range_and_prompt_unchanged(self):
        out = spuq_mod._perturb_temperature(QUESTION, 1.0, 10, np.random.default_rng(2))
        assert len(out) == 10
        for msgs, temp in out:
            assert msgs == QUESTION            # prompt untouched
            assert 0.0 <= temp <= 1.0          # jitter honoured (upstream arg-order bug fixed)

    def test_system_message_count_is_capped_at_available_messages(self):
        # only 9 canned system messages; asking for more must not crash or duplicate.
        out = spuq_mod._perturb_system_message(QUESTION, 1.0, 50, np.random.default_rng(3))
        assert len(out) == 9


class TestAggregation:
    def _spuq_with_fake_sims(self, output_sims, input_weights, weighted=True):
        """Build a SPUQ whose _sim / _input_weight return preset values, no rouge."""
        method = SPUQ(n_perturb=len(output_sims), perturbation="system_message",
                      aggregation="rougeL", weighted=weighted)
        # _run compares inp_out[1:] against inp_out[0]; provide sims/weights for those.
        sims = iter(output_sims)
        wts = iter(input_weights)
        method._sim = lambda a, b: next(sims)
        method._input_weight = lambda a, b: next(wts)
        return method

    def test_confident_case_high_agreement_high_confidence(self):
        m = self._spuq_with_fake_sims(output_sims=[1.0, 1.0, 1.0], input_weights=[1.0, 1.0, 1.0])
        inp_out = [(QUESTION, "Paris")] * 4
        assert m._aggregate(inp_out)["truth_value"] == pytest.approx(1.0)

    def test_fragile_case_low_agreement_low_confidence(self):
        m = self._spuq_with_fake_sims(output_sims=[0.0, 0.1, 0.0], input_weights=[1.0, 1.0, 1.0])
        inp_out = [(QUESTION, f"o{i}") for i in range(4)]
        conf = m._aggregate(inp_out)["truth_value"]
        assert conf == pytest.approx((0.0 + 0.1 + 0.0) / 3)

    def test_input_drift_downweights_a_sample(self):
        """The SPUQ-specific weighting: a perturbation that drifted far from the original
        input (low input weight) contributes less to the confidence."""
        # sample 1 agrees (sim 1.0) but its input barely drifted (weight 1.0);
        # sample 2 disagrees (sim 0.0) and its input drifted a lot (weight 0.1) -> discounted.
        m = self._spuq_with_fake_sims(output_sims=[1.0, 0.0], input_weights=[1.0, 0.1])
        conf = m._aggregate([(QUESTION, "a")] * 3)["truth_value"]
        # weighted mean = (1.0*1.0 + 0.0*0.1) / (1.0 + 0.1) = 1/1.1
        assert conf == pytest.approx(1.0 / 1.1)

    def test_unweighted_is_a_plain_mean(self):
        m = self._spuq_with_fake_sims(output_sims=[1.0, 0.0], input_weights=[999, 999],
                                      weighted=False)
        # weighted=False -> _input_weight returns 1.0 regardless (bypasses the fake)
        m._input_weight = lambda a, b: 1.0
        conf = m._aggregate([(QUESTION, "a")] * 3)["truth_value"]
        assert conf == pytest.approx(0.5)


class TestRunAndConfig:
    def test_run_uses_the_unperturbed_prompt_as_anchor_and_calls_target_per_perturbation(self):
        calls = []

        def fake_generate(msgs, temperature):
            calls.append((msgs, temperature))
            return "Paris"

        m = SPUQ(n_perturb=3, perturbation="system_message", aggregation="rougeL")
        m._sim = lambda a, b: 1.0
        m._input_weight = lambda a, b: 1.0
        out = m._run(fake_generate, QUESTION)
        # anchor (unperturbed) + 3 perturbations = 4 target calls
        assert len(calls) == 4
        assert calls[0][0] == QUESTION  # first call is the unperturbed prompt
        assert out["truth_value"] == pytest.approx(1.0)

    def test_num_target_calls_reflects_n_perturb(self):
        assert SPUQ(n_perturb=7).NUM_TARGET_CALLS == 7

    def test_is_not_a_shared_cache_method(self):
        """SPUQ perturbs the prompt, so it can't be served from the fixed-prompt cache."""
        assert SPUQ.REQUIRES_SAMPLED_TEXT is False

    def test_paraphrasing_requires_a_paraphrase_fn(self):
        with pytest.raises(ValueError, match="paraphrase_fn"):
            SPUQ(perturbation="paraphrasing")

    def test_rejects_unknown_perturbation_and_aggregation(self):
        with pytest.raises(ValueError, match="perturbation"):
            SPUQ(perturbation="nonsense")
        with pytest.raises(ValueError, match="aggregation"):
            SPUQ(aggregation="sbert")  # extra-dep metric, not in the text-only port
