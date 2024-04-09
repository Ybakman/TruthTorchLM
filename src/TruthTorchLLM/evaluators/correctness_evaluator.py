from abc import ABC, abstractmethod

class CorrectnessEvaluator(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, generated_text: str,  ground_truth_text: list[str]) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")