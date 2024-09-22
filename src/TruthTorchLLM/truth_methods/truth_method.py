from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from abc import ABC, abstractmethod
import torch
import random

class TruthMethod(ABC):

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False

    def __init__(self, threshold:float=0.0, std:float=1.0):
        self.threshold = threshold
        self.std = std
        

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)
    
    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)

    def get_threshold(self):
        return self.threshold
    
    def get_std(self):
        return self.std 

    def set_threshold(self, threshold:float):
        self.threshold = threshold
    
    def set_std(self, std:float):
        self.std = std
        
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
