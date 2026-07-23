"""Correctness evaluation for multiple-choice questions (benchmark D axis).

Upstream TruthTorchLM has no MCQ format, and MCQ is not just "free-form QA with a
shorter answer": the output space is constrained, so a model that says "B", "B)", "The
answer is B", or the full option text is right every time, while ``ExactMatch`` scores
three of those four as wrong. Under-counting correctness that way does not merely lower
the reported accuracy -- it corrupts the *labels* the UQ score is graded against, and the
protocol notes the correctness criterion materially moves AUROC.

The dataset loaders in ``TruthTorchLM.utils.dataset_utils`` put the option letter first
in ``ground_truths`` and the option text second, so both surface forms are accepted here.
"""

import re

from .correctness_evaluator import CorrectnessEvaluator

__all__ = ["MCQMatch"]

# "B", "(B)", "B.", "B)", "Answer: B", "The answer is B".
#
# Restricted to A-H (the option letters any of our MCQ sets use) and to *punctuated or
# standalone* leading letters. A bare "letter followed by a space" pattern looks
# harmless and is not: it reads "I am not sure about this one" as a commitment to
# option I, which would score an explicit non-answer as a confident one.
_OPTION_LETTER = "A-Ha-h"
_LETTER_PATTERNS = [
    re.compile(rf"^\s*\(([{_OPTION_LETTER}])\)"),            # (B) ...
    re.compile(rf"^\s*([{_OPTION_LETTER}])[\.\):,]"),        # B. / B) / B: / B,
    re.compile(rf"^\s*\(?([{_OPTION_LETTER}])\)?\s*$"),      # the whole answer is a letter
    re.compile(rf"\banswer\s*(?:is|:)?\s*\(?([{_OPTION_LETTER}])\)?\b", re.IGNORECASE),
    re.compile(rf"\boption\s*\(?([{_OPTION_LETTER}])\)?\b", re.IGNORECASE),
]


class MCQMatch(CorrectnessEvaluator):
    """Match a generation against an MCQ answer given as ``[letter, option_text]``.

    Deliberately conservative about what counts as a letter: a bare "B" anywhere in a
    sentence is *not* accepted, because "B" appears inside ordinary prose and a loose
    pattern would score wrong answers correct. Only the leading-token and explicit
    "answer is X" forms count.
    """

    def __init__(self, accept_option_text: bool = True):
        super().__init__()
        self.accept_option_text = accept_option_text

    @staticmethod
    def extract_letter(generated_text: str):
        """The option letter the generation commits to, or None if it commits to none."""
        text = (generated_text or "").strip()
        if not text:
            return None
        for pattern in _LETTER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).upper()
        return None

    def __call__(
        self,
        question_text: str,
        generated_text: str,
        ground_truths: list,
        context: str = "",
        seed: int = None,
    ) -> int:
        if not ground_truths:
            return -1  # TruthTorchLM's "no attempt / not scorable" sentinel

        gold_letter = str(ground_truths[0]).strip().upper()
        gold_text = str(ground_truths[1]).strip().lower() if len(ground_truths) > 1 else None

        predicted_letter = self.extract_letter(generated_text)
        if predicted_letter is not None:
            return int(predicted_letter == gold_letter)

        if self.accept_option_text and gold_text:
            generated = (generated_text or "").strip().lower()
            # Containment rather than equality: models routinely wrap the option text in
            # a sentence ("The most likely diagnosis is <option text>.").
            if generated == gold_text or gold_text in generated:
                return 1

        return 0

    def __str__(self):
        return "MCQ Match (option letter or option text)"
