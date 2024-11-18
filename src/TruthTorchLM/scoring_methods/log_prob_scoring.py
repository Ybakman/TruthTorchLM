from .scoring_method import ScoringMethod

class LogProbScoring(ScoringMethod):
    def __init__(self): 
        super().__init__()

    def __call__(self, logprobs: list[float]) -> float:
        return sum(logprobs)
    
    def  __str__(self):
        return "Log Prob Scoring"