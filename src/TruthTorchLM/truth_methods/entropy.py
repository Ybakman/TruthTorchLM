from .truth_method import TruthMethod
from TruthTorchLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np
from ..generation import sample_generations_hf_local, sample_generations_api


class Entropy(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True

    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations: int = 5, batch_generation = True):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.batch_generation = batch_generation


    def _entropy(self, generated_texts:list[str], question_context:str, scores:list[float]):
        entropy = -np.sum(scores) / len(scores)
        return {"truth_value": -entropy, 'entropy': entropy, "score_for_each_generation": scores, 'generated_texts': generated_texts}


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, batch_generation=self.batch_generation, **kwargs)
        
        scores = []
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
      
        for i in range(self.number_of_generations):
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale

        return self._entropy(generated_texts, question_context, scores)

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Entropy method is not applicable to given model")
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        scores = []
        for i in range(self.number_of_generations):
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale

        return self._entropy(generated_texts, question_context, scores)

    
