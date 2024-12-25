from abc import ABC, abstractmethod

class FactualDecompositionMethod(ABC):
    def __init__(self):
        pass

    def __call__(self, input_text)->list[str]:

        paragraphs = [paragraph.strip() for paragraph in input_text.split('\n') if paragraph.strip()]
        all_statements = []
        for paragraph in paragraphs:
            statements = self.decompose_facts(paragraph)
            for _ in range(self.decomposition_depth-1):
                temp_statements = []
                for statement in statements:
                    temp_statements.extend(self.decompose_facts(statement))
                statements = temp_statements
            all_statements.extend(statements)
        return all_statements

    @abstractmethod
    def decompose_facts(self, input_text:str)->list[str]:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")