from .correctness_evaluator import CorrectnessEvaluator
import evaluate

class ROUGE(CorrectnessEvaluator):
    def __init__(self, threshold: float = 0.5, rouge_type: str = 'rougeL'): 
        super().__init__()
        self.threshold = threshold  
        self.rouge = evaluate.load('rouge')
        self.rouge_type = rouge_type

    def __call__(self, question_text:str, generated_text: str,  ground_truths: list[str], seed:int = None) -> bool:
        for i in range(len(ground_truths)):
            rouge_results = self.rouge.compute(predictions = [generated_text], references=[ground_truths[i]])
            if rouge_results[self.rouge_type] > self.threshold:
                return 1
        return 0
    
    def __str__(self):
        return f"ROUGE with threshold {self.threshold} and type {self.rouge_type}"