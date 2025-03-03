from abc import ABC, abstractmethod


class DecompositionMethod(ABC):
    def __init__(self):

        pass

    def __call__(self, input_text) -> list[str]:

        return self.decompose_facts(input_text)

    @abstractmethod
    def decompose_facts(self, input_text: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
