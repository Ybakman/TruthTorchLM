"""Apply the §1 filter to the *real* TruthTorchLM methods.

Skipped when the full ML stack is unavailable, since it must import the actual classes.
Its job is to catch the single most consequential surprise in the shortlist: upstream
``SemanticEntropy`` requires sampled log-probabilities and is therefore **not** admissible
under the benchmark's text-only constraint. If a future upstream merge changes that flag,
this test tells us -- rather than a benchmark table quietly including a grey-box method.
"""

import pytest

pytest.importorskip(
    "torch", reason="requires the full TruthTorchLM dependency stack (torch/transformers/litellm)"
)

from TruthTorchLM.truth_methods import (  # noqa: E402
    DiscreteSemanticEntropy,
    NumSemanticSetUncertainty,
    SemanticEntropy,
    VerbalizedConfidence,
)
from TruthTorchLM.utils.access_level import AccessLevel, access_level  # noqa: E402


def test_semantic_entropy_is_grey_box_not_black_box():
    """The finding that reshapes the shortlist -- see protocol §1."""
    assert SemanticEntropy.REQUIRES_SAMPLED_LOGPROBS is True
    assert access_level(SemanticEntropy) is AccessLevel.GREY_BOX


def test_discrete_semantic_entropy_is_black_box():
    """DSE is the text-only SC workhorse that replaces grey-box SemanticEntropy in this
    regime (Farquhar et al. Nature 2024; jlko/semantic_uncertainty)."""
    assert DiscreteSemanticEntropy.REQUIRES_SAMPLED_LOGPROBS is False
    assert access_level(DiscreteSemanticEntropy) is AccessLevel.BLACK_BOX


def test_verbalized_confidence_is_black_box():
    assert access_level(VerbalizedConfidence) is AccessLevel.BLACK_BOX


def test_num_semantic_set_is_black_box():
    """The cluster-count consistency method needs sampled text only."""
    assert access_level(NumSemanticSetUncertainty) is AccessLevel.BLACK_BOX


def test_spuq_is_black_box_but_not_a_shared_cache_method():
    """SPUQ perturbs the prompt, so it makes its own target calls rather than reading the
    shared sample cache -- but it is still text-only, so black-box (Gao et al. EACL 2024)."""
    from TruthTorchLM.truth_methods import SPUQ

    assert access_level(SPUQ) is AccessLevel.BLACK_BOX
    assert SPUQ.REQUIRES_SAMPLED_TEXT is False
