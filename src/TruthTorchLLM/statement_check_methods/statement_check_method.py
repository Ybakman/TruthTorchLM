import torch
import random
from typing import Union
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class StatementCheckMethod(ABC):
    def __init__(self):
        pass


    def __call__(self, model:Union[PreTrainedModel, str], input_text:str = '', generated_text:str = '', question_context:str = '', 
                 statement:str='', text_so_far:str='', all_ids:Union[list, torch.Tensor] = None, 
                    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, 
                    messages:list = [], add_generation_prompt = True, continue_final_message = False, **kwargs) -> dict:
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)
        if isinstance(model, str):
            output_dict = self.check_statement_api(model=model, messages=messages, generated_text=generated_text, question_context=question_context, 
                                                   statement=statement, text_so_far=text_so_far, generation_seed=generation_seed, **kwargs)
        else:
            output_dict = self.check_statement_local(model=model, input_text=input_text, generated_text=generated_text, question_context=question_context, 
                                                     statement=statement, text_so_far=text_so_far, all_ids=all_ids, tokenizer=tokenizer, generation_seed=generation_seed, 
                                                     messages=messages, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message, **kwargs)

        return output_dict


    @abstractmethod
    def check_statement_local(self, model:PreTrainedModel, input_text:str, generated_text:str, 
                              question_context:str, statement:str, text_so_far:str, all_ids:Union[list, torch.Tensor], 
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed=None, 
                              messages:list = [], add_generation_prompt = True, continue_final_message = False, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def check_statement_api(self, model:str, messages:list, generated_text:str, 
                            question_context:str, statement:str, text_so_far:str, generation_seed=None, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")