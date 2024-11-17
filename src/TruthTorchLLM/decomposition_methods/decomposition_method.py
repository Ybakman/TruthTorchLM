from abc import ABC, abstractmethod

class FactualDecompositionMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def decompose_facts(self, input_text:str)->dict:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")