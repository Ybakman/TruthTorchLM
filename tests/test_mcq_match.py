"""MCQ correctness evaluation.

Why this needs its own evaluator rather than reusing ExactMatch: a model that answers
"B", "B)", "The answer is B", or the full option text is right in every case, and
ExactMatch scores three of the four wrong. That does not just depress reported accuracy
-- it flips the *labels* the UQ score is graded against, and the protocol notes the
correctness criterion materially moves AUROC.
"""

import pytest

pytest.importorskip("TruthTorchLM.evaluators.mcq_match", reason="needs the evaluators package")

from TruthTorchLM.evaluators.mcq_match import MCQMatch  # noqa: E402

GOLD = ["B", "Tell the attending that he cannot fail to disclose this mistake"]


@pytest.fixture
def evaluator():
    return MCQMatch()


@pytest.mark.parametrize(
    "generation",
    [
        "B",
        "B.",
        "B)",
        "(B)",
        " b ",
        "B. Tell the attending that he cannot fail to disclose this mistake",
        "The answer is B.",
        "Answer: B",
        "answer is (B)",
        "Option B",
    ],
)
def test_accepts_every_common_surface_form_of_the_right_letter(evaluator, generation):
    assert evaluator("q", generation, GOLD) == 1


@pytest.mark.parametrize("generation", ["A", "C.", "(D)", "The answer is A.", "Answer: C"])
def test_rejects_the_wrong_letter(evaluator, generation):
    assert evaluator("q", generation, GOLD) == 0


def test_accepts_the_option_text_without_a_letter(evaluator):
    assert evaluator("q", "Tell the attending that he cannot fail to disclose this mistake", GOLD) == 1


def test_accepts_option_text_wrapped_in_a_sentence(evaluator):
    generation = (
        "In this situation the right course of action is to tell the attending that he "
        "cannot fail to disclose this mistake."
    )
    assert evaluator("q", generation, GOLD) == 1


def test_a_stray_b_inside_prose_is_not_a_commitment(evaluator):
    """The reason the letter patterns are anchored rather than a bare \\bB\\b search:
    a loose pattern scores wrong answers as correct wherever an option letter happens to
    appear in ordinary English."""
    assert evaluator("q", "Both options seem plausible to me.", GOLD) == 0


def test_letter_wins_over_text_when_they_disagree(evaluator):
    """An explicit letter is the model's actual commitment. Quoting another option's text
    while choosing A must score as A -- otherwise a wrong answer that happens to restate
    the right option is credited."""
    generation = "A. Unlike the option to tell the attending that he cannot fail to disclose this mistake"
    assert evaluator("q", generation, GOLD) == 0


def test_empty_generation_is_wrong_not_an_error(evaluator):
    assert evaluator("q", "", GOLD) == 0


def test_missing_ground_truth_is_the_not_scorable_sentinel(evaluator):
    """TruthTorchLM uses -1 for 'not scorable', and metric_score filters those out."""
    assert evaluator("q", "B", []) == -1


def test_option_text_matching_can_be_disabled():
    strict = MCQMatch(accept_option_text=False)
    assert strict("q", GOLD[1], GOLD) == 0
    assert strict("q", "B", GOLD) == 1


def test_extract_letter_is_none_when_nothing_is_committed():
    assert MCQMatch.extract_letter("I am not sure about this one.") is None
    assert MCQMatch.extract_letter("") is None
