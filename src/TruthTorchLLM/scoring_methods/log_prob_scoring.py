from .scoring_method import ScoringMethod

class LogProbScoring(ScoringMethod):
    def __init__(self): 
        super().__init__()

    def __call__(self, input_text: str,  generated_tokens: list[str], logprobs: list[float], generated_text:str, generated_token_ids:list[int]=None) -> float:
        assert len(generated_tokens) == len(logprobs)
        return sum(logprobs)
    
    def  __str__(self):
        return "Log Prob Scoring"