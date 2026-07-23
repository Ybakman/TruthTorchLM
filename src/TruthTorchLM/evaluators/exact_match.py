from .correctness_evaluator import CorrectnessEvaluator

class ExactMatch(CorrectnessEvaluator):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        question_text: str,
        generated_text: str,
        ground_truths: list[str],
        context: str = "",
        seed: int = None,
    ) -> bool:
        # Upstream indexed the loop but compared against the whole list --
        # `ground_truths.strip()` raises AttributeError on a list, so the default
        # correctness evaluator crashed on its first call. Fixed to index the element.
        for i in range(len(ground_truths)):
            matched = generated_text.strip().lower() == str(ground_truths[i]).strip().lower()
            if matched:
                return 1
        return 0

    def __str__(self):
        return "Exact Match"