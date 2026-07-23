"""The pure black-box filter (benchmark protocol §1).

The filter must be mechanical, so these tests use plain stand-in classes carrying the
same ``REQUIRES_*`` flags as real TruthMethods. That keeps them runnable without the ML
stack, and it pins the *rule* rather than any one method's current flags.

``tests/test_black_box_shortlist.py`` applies the same rule to the real classes.
"""

import pytest

from TruthTorchLM.utils.access_level import (
    AccessLevel,
    access_level,
    access_report,
    filter_black_box,
    is_black_box,
)


class TextOnly:
    """A consistency method that needs sampled text and nothing else (e.g. discrete SE)."""

    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False
    REQUIRES_LOGPROBS = False
    number_of_generations = 5


class SinglePass:
    """A one-call verbalized method."""

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False
    REQUIRES_LOGPROBS = False


class NeedsLogprobs:
    """Upstream SemanticEntropy's shape: sampled text *and* sampled log-probabilities."""

    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False
    REQUIRES_LOGPROBS = False


class NeedsActivations:
    """An internal-state probe (e.g. SAPLMA / Inside)."""

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = True
    REQUIRES_LOGPROBS = False


def test_text_only_methods_are_black_box():
    assert access_level(TextOnly()) is AccessLevel.BLACK_BOX
    assert access_level(SinglePass()) is AccessLevel.BLACK_BOX
    assert is_black_box(TextOnly())


def test_sampling_alone_does_not_break_the_black_box_constraint():
    """N extra generations cost latency, not access. This is the Q1 axis, not the §1 one."""
    assert is_black_box(TextOnly())


def test_logprob_dependence_is_grey_box():
    assert access_level(NeedsLogprobs()) is AccessLevel.GREY_BOX
    assert not is_black_box(NeedsLogprobs())


def test_activation_dependence_is_white_box():
    assert access_level(NeedsActivations()) is AccessLevel.WHITE_BOX


def test_filter_strict_keeps_only_black_box():
    methods = [TextOnly(), SinglePass(), NeedsLogprobs(), NeedsActivations()]
    kept = filter_black_box(methods, strict=True)
    assert [type(m).__name__ for m in kept] == ["TextOnly", "SinglePass"]


def test_filter_non_strict_admits_grey_box_for_the_reference_line():
    methods = [TextOnly(), NeedsLogprobs(), NeedsActivations()]
    kept = filter_black_box(methods, strict=False)
    assert [type(m).__name__ for m in kept] == ["TextOnly", "NeedsLogprobs"]


def test_classes_work_as_well_as_instances():
    assert is_black_box(TextOnly)
    assert not is_black_box(NeedsLogprobs)


def test_report_rows():
    rows = access_report([TextOnly, NeedsLogprobs])
    assert rows[0] == {
        "method": "TextOnly",
        "access_level": "black-box",
        "requires_sampling": True,
        "number_of_generations": 5,
        "num_target_calls": 1,
    }
    assert rows[1]["access_level"] == "grey-box"
