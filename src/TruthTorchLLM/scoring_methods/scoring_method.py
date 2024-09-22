from abc import ABC, abstractmethod

class ScoringMethod(ABC):
    #Scoring methods should be implemented as subclasses of this class
    #forward method should be implemented in subclasses
    
    def __init__(self):
        pass

    @abstractmethod
    #tokens: list of tokens in the generated text
    #logprobs: list of log probabilities of each token in the generated text
    #returns a float score
    def __call__(self, input_text: str,  generated_tokens: list[str], logprobs: list[float], generated_text:str, generated_token_ids:list[int]=None) -> float:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")