"""IUQ — the ported scoring math, parsers, and the generation-level pipeline.

IUQ is a long-form, LLM-heavy pipeline; the LLM calls need the stack + a model. What is
porting-sensitive and tested here is the exact scoring math (impact with exp-decay error
propagation, faithfulness, IUQ = supportness x impact) against hand-computed values, the
judge-reply parsers, and the pipeline wiring via an injected `chat` callable. The module is
pure-numpy, so it loads standalone.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _load():
    dotted = "TruthTorchLM.long_form_generation.iuq"
    if dotted in sys.modules:
        return sys.modules[dotted]
    import types

    for pkg in ("TruthTorchLM", "TruthTorchLM.long_form_generation"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(
        dotted, SRC / "TruthTorchLM/long_form_generation/iuq.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


iuq = _load()


class TestImpactMath:
    def test_without_error_propagation_is_one_minus_contradiction(self):
        got = iuq.compute_impacts([0.0, 0.5, 1.0], with_error_propagation=False)
        assert np.allclose(got, [1.0, 0.5, 0.0])

    def test_error_propagation_matches_the_source_formula(self):
        """Verbatim recomputation of gather_impacts: exp(-convolve(c, exp(-arange))[:n])."""
        contradictions = [0.2, 0.8, 0.1, 0.5]
        c = np.array(contradictions)
        weight_func = np.exp(-np.arange(len(c)))
        weights = np.convolve(c, weight_func)[: len(c)]
        expected = 1 / np.exp(weights)
        got = iuq.compute_impacts(contradictions, with_error_propagation=True)
        assert np.allclose(got, expected)

    def test_no_contradiction_gives_impact_one(self):
        got = iuq.compute_impacts([0.0, 0.0, 0.0], with_error_propagation=True)
        assert np.allclose(got, 1.0)

    def test_a_contradicted_claim_lowers_downstream_impact(self):
        """Error propagation: a highly-contradicted early claim drags down later ones."""
        clean = iuq.compute_impacts([0.0, 0.0, 0.0])
        contaminated = iuq.compute_impacts([0.9, 0.0, 0.0])
        # the later claims are lower when an upstream claim was contradicted
        assert contaminated[1] < clean[1]

    def test_empty_is_empty(self):
        assert iuq.compute_impacts([]).size == 0


class TestFaithfulnessAndIUQ:
    def test_faithfulness_is_one_minus_contradiction(self):
        assert np.allclose(iuq.compute_faithfulness([0.1, 0.4]), [0.9, 0.6])

    def test_iuq_is_supportness_times_impact(self):
        assert iuq.iuq_score(0.8, 0.5) == pytest.approx(0.4)

    def test_iuq_none_supportness_propagates(self):
        assert iuq.iuq_score(None, 0.5) is None

    def test_more_supported_and_faithful_scores_higher(self):
        strong = iuq.iuq_score(1.0, 1.0)      # every generation supports it, no contradiction
        weak = iuq.iuq_score(0.2, 0.3)         # few support it, contradicted
        assert strong > weak


class TestParsers:
    @pytest.mark.parametrize("reply,expected", [
        ("30", 0.30), ("The answer is 45%.", 0.45), ("0", 0.0), ("100", 1.0),
        ("I would change about 20 percent", 0.20),
    ])
    def test_contradiction_percentage(self, reply, expected):
        assert iuq.parse_contradiction_percentage(reply) == pytest.approx(expected)

    def test_contradiction_out_of_range_or_unparseable_is_zero(self):
        assert iuq.parse_contradiction_percentage("no number here") == 0.0
        assert iuq.parse_contradiction_percentage("140") == 0.0

    def test_contradiction_uses_the_last_sentence(self):
        # mirrors the source: split on sentence enders, take the last
        assert iuq.parse_contradiction_percentage("Let me think. 60") == pytest.approx(0.60)

    @pytest.mark.parametrize("reply", ["True", "true", "Yes, supported", "YES"])
    def test_support_vote_true(self, reply):
        assert iuq.parse_support_vote(reply) is True

    @pytest.mark.parametrize("reply", ["False", "no", "not supported", ""])
    def test_support_vote_false(self, reply):
        assert iuq.parse_support_vote(reply) is False


class TestPipelineWiring:
    def test_score_generation_combines_supportness_and_impact_via_injected_chat(self):
        """End-to-end on the pure path: a scripted `chat` drives every LLM stage, so the
        combination (supportness vote + contradiction -> impact -> IUQ) is exact and
        model-free."""
        # chat routes by prompt content: support judge -> "True", contradiction -> "0",
        # question gen / answer -> a dummy string.
        def chat(messages, temperature):
            text = messages[-1]["content"]
            if "supported by the given passage" in text:
                return "True"          # every generation supports every claim -> supportness 1.0
            if "how much of the context" in text:
                return "0"             # no contradiction -> impact 1.0
            return "some text"

        method = iuq.InterrogativeUQ(chat=chat, n_answers=2)
        results = method.score_generation(
            claims=["Claim A", "Claim B"],
            diverse_generations=["gen1", "gen2", "gen3"],
        )
        assert len(results) == 2
        for r in results:
            assert r["supportness_score"] == pytest.approx(1.0)
            assert r["impact"] == pytest.approx(1.0)
            assert r["truth_value"] == pytest.approx(1.0)   # IUQ = 1.0 * 1.0

    def test_contradicted_unsupported_claim_scores_low(self):
        def chat(messages, temperature):
            text = messages[-1]["content"]
            if "supported by the given passage" in text:
                return "False"         # unsupported -> supportness 0.0
            if "how much of the context" in text:
                return "80"            # heavy contradiction
            return "some text"

        method = iuq.InterrogativeUQ(chat=chat, n_answers=1)
        results = method.score_generation(claims=["Shaky claim"], diverse_generations=["g1", "g2"])
        assert results[0]["supportness_score"] == pytest.approx(0.0)
        assert results[0]["truth_value"] == pytest.approx(0.0)  # 0 supportness zeroes IUQ
