
from .correctness_evaluator import CorrectnessEvaluator
import evaluate


class BLEU(CorrectnessEvaluator):
    def __init__(self, threshold: float = 0.5): 
        super().__init__()
        self.threshold = threshold  
        self.bleu = evaluate.load('bleu')

    def __call__(self, question_text:str, generated_text: str,  ground_truths: list[str], seed:int = None) -> bool:
        for i in range(len(ground_truths)):
            bleu_results = self.bleu.compute(predictions = [generated_text], references=[ground_truths[i]])
            if bleu_results['bleu'] > self.threshold:
                return 1
        return 0
    
    def __str__(self):
        return f"BLEU with threshold {self.threshold}"