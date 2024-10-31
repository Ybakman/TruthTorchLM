from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from abc import ABC, abstractmethod
import torch
import random
from TruthTorchLLM.utils import sigmoid_normalization

class TruthMethod(ABC):

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False
    REQUIRES_NORMALIZATION = True

    def __init__(self, threshold:float=0.0, std:float=1.0):
        self.threshold = threshold
        self.std = std
        
    def __call__(self, model:Union[PreTrainedModel, str], input_text:str = '', generated_text:str = '', question_context:str = '', all_ids:Union[list, torch.Tensor] = None, 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)
        if isinstance(model, str):
            output_dict = self.forward_api(model=model, messages=messages, generated_text=generated_text, question_context=question_context, generation_seed=generation_seed, sampled_generations_dict=sampled_generations_dict, **kwargs)
        else:
            output_dict = self.forward_hf_local(model=model, input_text=input_text, generated_text=generated_text, question_context=question_context, all_ids=all_ids, 
            tokenizer=tokenizer, generation_seed=generation_seed, sampled_generations_dict=sampled_generations_dict, messages=messages, **kwargs)
        
        if self.REQUIRES_NORMALIZATION:
            output_dict['normalized_truth_value'] = self.normalize(output_dict['truth_value'])
        else:
            output_dict['normalized_truth_value'] = output_dict['truth_value']  
        return output_dict

    @abstractmethod
    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def normalize(self, truth_value:float):
        return sigmoid_normalization(truth_value, self.threshold, self.std)
        

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
    
