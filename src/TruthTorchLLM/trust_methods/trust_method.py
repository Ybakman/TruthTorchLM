from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from abc import ABC, abstractmethod

class TrustMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, model:Union[str,PreTrainedModel], input_text:str, generated_text:str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
