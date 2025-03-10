import torch
import random
from typing import Union
from abc import ABC, abstractmethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class ClaimCheckMethod(ABC):
    def __init__(self):
        pass

    def __call__(
        self,
        model: Union[PreTrainedModel, str],
        input_text: str = "",
        generated_text: str = "",
        question: str = "",
        claim: str = "",
        text_so_far: str = "",
        all_ids: Union[list, torch.Tensor] = None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        messages: list = [],
        context:str = "",
        **kwargs
    ) -> dict:
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            random.seed(generation_seed)
        if isinstance(model, str):
            output_dict = self.check_claim_api(
                model=model,
                messages=messages,
                generated_text=generated_text,
                question=question,
                claim=claim,
                text_so_far=text_so_far,
                generation_seed=generation_seed,
                context=context,
                **kwargs
            )
        else:
            output_dict = self.check_claim_local(
                model=model,
                input_text=input_text,
                generated_text=generated_text,
                question=question,
                claim=claim,
                text_so_far=text_so_far,
                all_ids=all_ids,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                messages=messages,
                context=context,
                **kwargs
            )

        return output_dict

    @abstractmethod
    def check_claim_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        claim: str,
        text_so_far: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        messages: list = [],
        context:str = "",
        **kwargs
    ) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def check_claim_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        claim: str,
        text_so_far: str,
        generation_seed=None,
        context:str = "",
        **kwargs
    ) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Subclasses must implement this method")
