import torch
import random
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat
from TruthTorchLM.normalizers import Normalizer, SigmoidNormalizer
import litellm
litellm.drop_params = True


def sigmoid_normalization(x: float, threshold: float = 0.0, std: float = 1.0):
    z = (x - threshold) / std
    if z >= 0:
        # For positive z, compute sigmoid as 1 / (1 + exp(-z)) directly
        return 1 / (1 + np.exp(-z))
    else:
        # For negative z, to avoid overflow, use the identity: sigmoid(z) = exp(z) / (1 + exp(z))
        return np.exp(z) / (1 + np.exp(z))

class TruthMethod(ABC):

    REQUIRES_SAMPLED_TEXT = False
    REQUIRES_SAMPLED_LOGITS = False
    REQUIRES_SAMPLED_LOGPROBS = False
    REQUIRES_SAMPLED_ATTENTIONS = False
    REQUIRES_SAMPLED_ACTIVATIONS = False
    REQUIRES_NORMALIZATION = True
    REQUIRES_LOGPROBS = False

    def __init__(self):
        self.normalizer = SigmoidNormalizer(threshold = 0, std = 1.0)#default dummy normalizer

    def __call__(self, model:Union[PreTrainedModel, str], input_text:str = '', generated_text:str = '', question_context:str = '', all_ids:Union[list, torch.Tensor] = None, 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], logprobs:list=None, generated_tokens:list=None, **kwargs):
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)
        if isinstance(model, str):
            output_dict = self.forward_api(model=model, messages=messages, generated_text=generated_text, question_context=question_context, 
            generation_seed=generation_seed, sampled_generations_dict=sampled_generations_dict, logprobs=logprobs, generated_tokens=generated_tokens, **kwargs)
        else:
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            output_dict = self.forward_hf_local(model=model, input_text=input_text, generated_text=generated_text, question_context=question_context, all_ids=all_ids, 
            tokenizer=tokenizer, generation_seed=generation_seed, sampled_generations_dict=sampled_generations_dict, messages=messages, **kwargs)
        
        if self.REQUIRES_NORMALIZATION:
            output_dict['normalized_truth_value'] = self.normalizer(output_dict['truth_value'])
        else:
            output_dict['normalized_truth_value'] = output_dict['truth_value']  
        return output_dict

    @abstractmethod
    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def set_normalizer(self, normalizer:Normalizer):
        self.normalizer = normalizer
    
    def get_normalizer(self):
        return self.normalizer
        
    
    def __str__(self):
        #search over all attributes and print them
        return f"{self.__class__.__name__} with {str(self.__dict__)}"
    
