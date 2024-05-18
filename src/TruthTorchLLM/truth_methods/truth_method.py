from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from abc import ABC, abstractmethod
import torch

class TruthMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def completion_forward(self, model:str, input_text:str, generated_text:str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
