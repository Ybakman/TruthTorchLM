from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from abc import ABC, abstractmethod
import torch


class StatementCheckMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check_statement_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, statement:str, text_so_far:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def check_statement_api(self, model:str, messages:list, generated_text:str, question_context:str, statement:str, text_so_far:str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")