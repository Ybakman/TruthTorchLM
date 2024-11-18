from .scoring_method import ScoringMethod


class LengthNormalizedScoring(ScoringMethod):
    def __init__(self): 
        super().__init__()

    def __call__(self, logprobs: list[float]) -> float:
        return sum(logprobs) / len(logprobs)
    
    def __str__(self):
        return "Length Normalized Scoring"