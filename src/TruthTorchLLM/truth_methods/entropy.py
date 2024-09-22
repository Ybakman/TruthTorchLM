from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np
import copy
import random
from ..generation import sample_generations_batch_hf_local, sample_generations_sequential_hf_local, sample_generations_api


class Entropy(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True

    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations: int = 5, threshold:float = 0.0, std:float = 1.0):#normalization, 
        super().__init__(threshold = threshold, std = std)
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations


    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_sequential_hf_local(model, input_text, tokenizer, [self], generation_seed, **kwargs)
        
        scores = []
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
      
        for i in range(self.number_of_generations):
            tokens_text = [tokenizer.decode(token) for token in sampled_generations_dict["tokens"][i]]
            score = self.scoring_function(question_context, tokens_text, sampled_generations_dict["logprobs"][i], generated_texts[i], sampled_generations_dict["tokens"][i]) 
            scores.append(score) #scores are in log scale

        entropy = -np.sum(scores) / len(scores)#scores are in log scale

        normalized_truth_value = sigmoid_normalization(-entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_truth_value':normalized_truth_value,  'entropy': entropy,  "score_for_each_generation": scores, 'generated_texts_for_entropy': generated_texts}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Entropy method is not applicable to given model")
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model, messages, [self], generation_seed, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        scores = []
        for i in range(self.number_of_generations):
            score = self.scoring_function(question_context, sampled_generations_dict["tokens"][i], sampled_generations_dict["logprobs"][i], generated_texts[i]) 
            scores.append(score) #scores are in log scale

        entropy = -np.sum(scores) / len(scores)#scores are in log scale

        normalized_truth_value = sigmoid_normalization(-entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_truth_value':normalized_truth_value,  'entropy': entropy,  "score_for_each_generation": scores, 'generated_texts_for_entropy': generated_texts}#this output format should be same for all truth methods

    
    def __str__(self):
        return "Entropy Truth Method with " + str(self.number_of_generations) + " generations. Scoring function: " + str(self.scoring_function) + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)