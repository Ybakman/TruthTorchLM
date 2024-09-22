import numpy as np
from .scoring_method import ScoringMethod
from sentence_transformers import CrossEncoder
from transformers import PreTrainedTokenizer


class TokenSAR(ScoringMethod):
    def __init__(self, tokenizer:PreTrainedTokenizer, similarity_model=None, similarity_model_device = 'cuda'): 
        super().__init__()

        self.tokenizr = tokenizer
        if similarity_model is None:
            self.similarity_model = CrossEncoder('cross-encoder/stsb-roberta-large', device=similarity_model_device)
        else:
            self.similarity_model = similarity_model

    def __call__(self, input_text: str,  generated_tokens: list[str], logprobs: list[float], generated_text:str, generated_token_ids:list[int]=None) -> float:
        assert len(generated_tokens) == len(logprobs)

        importance_vector = []
        for i in range(len(generated_token_ids)):
            removed_answer_ids = generated_token_ids[:i] + generated_token_ids[i+1:]
            removed_answer = self.tokenizer.decode(removed_answer_ids , skip_special_tokens=True)
            score = self.similarity_model.predict([( input_text +" "+removed_answer, input_text + ' ' + generated_text)])
            score = 1 - score[0]
            importance_vector.append(score)

        importance_vector = importance_vector / np.sum(importance_vector)
        return np.dot(importance_vector, logprobs)
    
    def __str__(self):
        return f"TokenSAR with tokenizer {self.tokenizr} and similarity model {self.similarity_model.config._name_or_path}"