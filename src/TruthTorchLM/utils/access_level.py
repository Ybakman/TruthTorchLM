"""The pure black-box filter (benchmark protocol §1).

The benchmark's scope constraint is that a method must run on the target's *generated
text only* -- no weights, no hidden states, no token log-probabilities. That is the
regime an inline guardrail wrapping an external coaching model actually lives in.

TruthTorchLM already encodes exactly the information needed to decide this, as the
``REQUIRES_*`` class flags on ``TruthMethod``. So the filter is mechanical rather than
editorial, which matters: it means no method gets into the shortlist by an author's
say-so, and surprises surface on their own.

One such surprise is worth stating up front: upstream ``SemanticEntropy`` sets
``REQUIRES_SAMPLED_LOGPROBS = True`` and is therefore **grey-box, not black-box**. The
consistency workhorse for our regime has to be the discrete / cluster-assignment
variant, which needs sampled text only.
"""

from enum import Enum

__all__ = ["AccessLevel", "access_level", "is_black_box", "filter_black_box", "access_report"]


class AccessLevel(str, Enum):
    """How much of the target model a method needs to see."""

    BLACK_BOX = "black-box"  # generated text only
    GREY_BOX = "grey-box"  # + token probabilities / logits
    WHITE_BOX = "white-box"  # + hidden states / attentions


#: Flags that push a method past text-only. Ordered most- to least-invasive.
_WHITE_BOX_FLAGS = ("REQUIRES_SAMPLED_ACTIVATIONS", "REQUIRES_SAMPLED_ATTENTIONS")
_GREY_BOX_FLAGS = ("REQUIRES_SAMPLED_LOGITS", "REQUIRES_SAMPLED_LOGPROBS", "REQUIRES_LOGPROBS")


def _flag(method, name: bool) -> bool:
    return bool(getattr(method, name, False))


def access_level(method) -> AccessLevel:
    """Classify a ``TruthMethod`` (instance or class) by the access it requires."""
    if any(_flag(method, f) for f in _WHITE_BOX_FLAGS):
        return AccessLevel.WHITE_BOX
    if any(_flag(method, f) for f in _GREY_BOX_FLAGS):
        return AccessLevel.GREY_BOX
    return AccessLevel.BLACK_BOX


def is_black_box(method) -> bool:
    """True if the method scores using generated text only."""
    return access_level(method) is AccessLevel.BLACK_BOX


def filter_black_box(methods: list, strict: bool = True) -> list:
    """Keep only the methods admissible under the benchmark's scope constraint.

    With ``strict=False`` grey-box methods are kept too, which is how the white-box
    reference line -- clearly labelled *not available in deployment* -- gets built.
    """
    if strict:
        return [m for m in methods if is_black_box(m)]
    return [m for m in methods if access_level(m) is not AccessLevel.WHITE_BOX]


def access_report(methods: list) -> list:
    """Per-method (name, access level, sampling requirement) rows, for the spec table.

    ``requires_sampling`` distinguishes the single-pass floor from the O(N) consistency
    family -- the other axis the protocol crosses.
    """
    rows = []
    for m in methods:
        name = m.__name__ if isinstance(m, type) else type(m).__name__
        rows.append(
            {
                "method": name,
                "access_level": access_level(m).value,
                "requires_sampling": _flag(m, "REQUIRES_SAMPLED_TEXT")
                or _flag(m, "REQUIRES_SAMPLED_LOGPROBS")
                or _flag(m, "REQUIRES_SAMPLED_LOGITS"),
                "number_of_generations": getattr(m, "number_of_generations", 1),
                "num_target_calls": getattr(m, "NUM_TARGET_CALLS", 1),
            }
        )
    return rows
